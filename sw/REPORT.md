# CIMple software reproduction: report

Paper: Ahn et al., *CIMple: Standard-cell SRAM-based CIM with LUT-based split
softmax for attention acceleration*, arXiv:2604.15944v2.
Package: `sw/` (`cimple_sw`). How to run each piece is in `README.md`. This
file records **what came out** and how it compares with the paper.

Status legend: ✅ reproduced, ⚠️ reproduced with caveats, ❌ not reproduced. All runs are complete.

## 1. Summary

| Paper claim | Here | Status |
|---|---|---|
| LUT-based split softmax in fixed point, INT8 (Sec. IV-B, Fig. 7) | `split_softmax.py`. The 8b-table mode is bit-exact with the RTL golden model `gm.softmax_row` | ✅ |
| INT8 TinyLlama + lm-eval: ARC-C / HellaSwag / OBQA / Winogrande within ±0.6 of the PyTorch-softmax baseline (Fig. 11) | `quant.py`, `hf_attention.py`, `eval.py`. Deltas of +0.94 / +1.40 / -0.79 on ARC-C / OBQA / Winogrande (full splits), all inside one stderr. HellaSwag (first 2000 docs) -0.25 vs the paper's -0.54. Details in §3 | ✅ |
| Split softmax cuts activation-to-activation latency by 33% (Sec. V-B) | `latency.py`: **42.8%** with first-principles defaults; 33-34% under several single documented baseline choices | ⚠️ |
| Encoder-only / decoder-only / encoder-decoder mappings (Sec. IV-C/D/E) | `mapping.py`: bit-exact, cycle-annotated, all three mappings | ✅ |
| 26.1 TOPS/W, 2.31 TOPS/mm² | Needs 28 nm silicon/layout, so software can't reproduce it | n/a |

Tests: `uv run pytest -q` gives **285 passed**, including the slow TinyLlama
test.

## 2. Split-softmax numerics

- **8b tables (`cimple_hw8`).** The kernel matches the golden model bit for
  bit: e, exp-sum, e·V partial and the INT8 output are compared with `==`
  over many rows, scales, lengths and masks.
- **Full-precision tables (`cimple`, the configuration the paper
  evaluates).** On random attention (score std 2-5, L 64-256) the cosine
  against float attention is 0.9993-0.9997.
- **8b tables with a constant z_max = 127 are fragile.** A row whose max score
  sits well below 127 rounds every exp entry to 0, so it outputs 0:

  | case | cosine vs float | rows with exp-sum 0 |
  |---|---|---|
  | hw8, score std 2 | 0.908 | 3.4% |
  | hw8, score std 5 | 0.515 | 79.6% |
  | hw8 + per-row max (ablation) | 0.999 | 0% |

  That is why the paper's accuracy study uses full-precision tables. It is
  also a real design gap for 8b hardware tables (see `docs/01-spec-teardown.md`
  §3 and roadmap "research" item 1).
- **The absmax score scale makes it worse.** If a head's negative tail is
  larger than its positive max, the whole head can underflow under hw8. A
  scale set from the positive max fixes that.

## 3. Accuracy study (TinyLlama-1.1B-Chat-v1.0, W8A8, 0-shot)

Setup: weights INT8 per output channel, activations INT8 per token (fake
quant), lm_head kept in fp. The metric is acc_norm for ARC-C, HellaSwag and
OBQA and acc for Winogrande. That choice is [A]: the paper doesn't say which
metric it reports.

Presets:

- `float`: the paper's baseline (stock softmax).
- `float_int8attn`: the same INT8 Q/K/V/output as `cimple`, with an fp32
  softmax. `cimple` minus `float_int8attn` isolates the softmax approximation.
- `cimple`: the paper's evaluated configuration.

Paper (Fig. 11), CIMple / delta vs baseline: ARC-C 31.74 / +0.6, HellaSwag
60.52 / -0.54, OBQA 36.8 / -0.2, Winogrande 57.22 / -0.32.

### 3a. Main study: full ARC-C / OBQA / Winogrande, HellaSwag first 2000 docs

Source: `results/main3/summary.md`, full validation/test splits. Stderr is
about ±1.4 for ARC-C and Winogrande and ±2.1 for OBQA.

| task (metric) | float | float_int8attn | **cimple** | delta vs float | delta vs float_int8attn (softmax only) | paper CIMple | paper delta |
|---|---|---|---|---|---|---|---|
| ARC-C (acc_norm) | 32.42 | 32.94 | **33.36** | +0.94 | +0.43 | 31.74 | +0.6 |
| OBQA (acc_norm) | 35.20 | 36.20 | **36.60** | +1.40 | +0.40 | 36.8 | -0.2 |
| Winogrande (acc) | 60.22 | 60.62 | **59.43** | -0.79 | -1.18 | 57.22 | -0.32 |
| HellaSwag (acc_norm, first 2000 docs) | 55.85 | n/a | **55.60** | -0.25 | n/a | 60.52 | -0.54 |

What this shows:

- **The paper's claim reproduces.** CIMple's full-precision LUT split
  softmax on an INT8 TinyLlama stays within the noise of the PyTorch-softmax
  baseline. Every delta is ≤ 1.4 points and inside one stderr. The paper's
  ±0.6 also sits inside our error bars.
- **The sign pattern matches:**
  - ARC-C improves slightly; the paper sees the same.
  - Winogrande drops slightly; the paper sees the same.
  - OBQA is +0.4 here (softmax only) where the paper has -0.2. Both are
    noise.
- **HellaSwag is a subset.** Absolute HellaSwag is lower than the paper's
  60.52 because it covers only the first 2000 of 10042 docs (stderr ±1.1).
  The paired delta of -0.25 matches the paper's -0.54 in sign and size.
- **The absolute numbers are close to the paper's.** ARC-C 33.4 vs 31.7 and
  OBQA 36.6 vs 36.8. Winogrande is 2.2 points higher here. That points to a
  slightly different checkpoint, INT8 scheme or harness version (all [A]),
  not to the softmax.
- **`float_int8attn` tells the two effects apart.** INT8 quantisation of
  Q, K, V and the attention output alone moves accuracy by -0.9 to +1.0. The
  LUT softmax adds +0.4 / +0.4 / -1.2 on top.

### 3b. Hardware-table ablation: 250 docs per task (stderr about ±3 points)

Source: `results/ablation250/summary.md`. A positive delta means better than
`float`. Paper delta is for its full-precision configuration.

| task (metric) | float | cimple_hw8 (8b exp LUT, const z_max) | cimple_hw8_rowmax (8b exp LUT, per-row max) | paper delta |
|---|---|---|---|---|
| ARC-C (acc_norm) | 34.40 | 25.20 (**-9.2**) | 32.40 (-2.0) | +0.6 |
| HellaSwag (acc_norm) | 57.60 | 32.40 (**-25.2**) | 56.00 (-1.6) | -0.54 |
| OBQA (acc_norm) | 37.60 | 32.40 (**-5.2**) | 36.80 (-0.8) | -0.2 |
| Winogrande (acc) | 59.60 | 48.80 (**-10.8**) | 58.00 (-1.6) | -0.32 |

What this shows:

- **The 8b exp table with the paper's constant z_max collapses on a real
  LLM.** It loses 5-25 points. TinyLlama's attention rows often peak well
  below the head's absmax, so every entry of such a row rounds to 0.
- **The constant max is what fails, not the 8b table.** Swapping in a
  per-row max while keeping the same 8b tables brings every task back to
  within 0.8-2.0 points. That is about the size of the stderr at 250 docs.
- **The paper's ±0.6 claim depends on full-precision tables.** It
  evaluated with them, and §3a tests that configuration.
- **What an 8b-table chip needs.** Some form of max (per row, or a coarse
  per-tile max) or a different scale policy. This is the research angle
  noted in `docs/02-roadmap.md`.

## 4. Latency (Sec. V-B): N = 1024 tokens, head dim 64, 400 MHz, encoder-only

| variant | cycles | time | GOPS | core util |
|---|---|---|---|---|
| baseline (non-split LUT softmax, 32b input, 3 reads) | 1,835,362 | 4.59 ms | 58.5 | 28.6% |
| split (CIMple) | 1,049,730 | 2.62 ms | 102.3 | 49.9% |
| **reduction** | | **42.8%** | | (paper: 33%) |

- **Why 42.8%.** The baseline adds three passes over the 32 × 1024 score tile
  (max, exp-sum, normalise), each N²/4 cycles at 4 lanes. Split mode hides all
  of that behind the CIM core. Roughly, reduction ≈ 24N / 56N. The result is
  flat in N for N ≥ 256 and depends on d: 49.9% at d = 32, 42.8% at d = 64,
  27.2% at d = 128.
- **Not tuned to 33%.** Single, documented choices for the unspecified
  baseline do land near 33%:
  - computing the max on the fly, so 2 reads instead of 3: 33.3%
  - streaming the normalised values straight into A'V: 34.0%
  - `overlap_out` + 1 read: 32.7%
- **The number hinges on the baseline.** A baseline that overlaps the softmax
  of Q-tile i with the QK^T of tile i+1 gains almost nothing (1.4%). The 33%
  claim is therefore only as strong as the unstated baseline.

## 5. Mapping simulator (Sec. IV-C/D/E)

At d_model 256, 4 heads, head dim 64, with W_O, 400 MHz:

| mapping | cycles | utilisation of 512 OP/cycle |
|---|---|---|
| encoder layer, N = 128 | 328,064 (820 µs) | 49.9% (capped at 50% by the 8+8-cycle pass) |
| decoder step, context 128 | 20,528 (51 µs) | 6.2% (write-bound: weight/KV tiles are re-written into the banks every token) |
| encoder-decoder | combined in `cimple-sw map` | see output |

- Every mapping equals an independent integer reference bit for bit.
- Decoder token-by-token with a KV cache equals the causal encoder bit for
  bit, given static scales.
- The attention-phase cycles match `latency.py` to within 0.1%.

## 6. Findings worth acting on (outside `sw/`, left untouched)

1. **Rounding bias in the requantiser.** `gm.quantise` (golden model) and
   `rtl/quant_unit.sv` (lines 30-37, checked) compute `floor(x - 1/2)` for negative
   values: `prod - 2^(sh-1)` followed by an arithmetic shift. That is
   1 LSB too negative on every negative non-tie, not round-half-away.
   - Example: −5/4 gives −2, but round-half-away gives −1.
   - The fix is to negate, round, and negate back (or add `2^(sh-1) - 1`
     for negative values before the shift).
   - `sw` mirrors the current behaviour so that hw8 stays bit-exact. A
     regression test documents it
     (`test_requantise_negative_rounding_is_floor_minus_half`).
2. **Unsigned exp values vs signed activations.** e from the exp LUT is
   UQ0.8 (0..255), but the CIM treats activations as signed INT8.
   `mapping.py` defaults to `e >> 1` (your nearmem decision 4). Measured
   against an exact softmax over the same INT8 scores:
   - `e >> 1`: mean error 4.5 / 1.2 / 1.3 LSB, worst 77 LSB
   - unsigned-activation mode (the sign cycle adds; an RTL change): mean
     error 1.2 / 0.7 / 0.7 LSB
3. **Constant z_max with 8b tables** (§2). For the research angle in the
   roadmap, the per-row max ablation (`cimple_hw8_rowmax`) recovers almost
   everything. A coarse per-tile max is the cheap hardware version to try.
4. **The paper contradicts itself on where e goes for encoder-only.**
   - Sec. IV-A says the softmax output goes "to the write data".
   - Sec. IV-C says V is stationary and e is fed as input.
   - `latency.py` and `mapping.py` follow IV-C and document the other
     reading (it costs 64k extra write cycles at N = 1024).
