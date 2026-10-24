# cimple-sw

Software side of the CIMple re-implementation:

> Ahn et al., 2025. *CIMple: Standard-cell SRAM-based CIM with LUT-based split
> softmax for attention acceleration*, arXiv:2604.15944v2.

The RTL, cocotb testbenches and the bit-exact numpy golden model live in the
parent project (`../src/cimple_rtl`, package `cimple_rtl`). This package depends
on it as an editable path dependency. All integer behaviour (exp/reciprocal LUTs,
`quantise`, `softmax_row`, `to_int8`) comes from the golden model or is tested
against it with `==`.

## What is reproduced, and where

| Paper claim / section | Module | Command | Status |
|---|---|---|---|
| LUT-based split softmax (Sec. IV-B, Fig. 7): exp LUT for the numerator, exp-sum and reciprocal applied *after* `e @ V` | `cimple_sw/split_softmax.py`, `cimple_sw/luts.py` | `uv run pytest tests/test_split_softmax.py` | Bit-exact with `gm.softmax_row` for the `hw8` tables (e, sum, partial, INT8 out) |
| 256-entry exp / reciprocal LUT contents for a score scale | `cimple_sw/luts.py` | `uv run cimple-sw luts --score-scale 0.05 [--precision hw8\|full] [--out-dir DIR]` | `--precision hw8` (default): `$readmemh` hex (`exp_lut.hex`, `recip_lut.hex`), same tables as `gm.build_exp_lut` / `gm.build_recip_lut`. `--precision full`: the full-precision exp table as decimal text in `exp_lut_full.txt`, which `$readmemh` cannot load and the RTL does not use |
| Accuracy on INT8 TinyLlama with lm-evaluation-harness: ARC-C 31.74, HellaSwag 60.52, OBQA 36.8, Winogrande 57.22 (deltas +0.6 / -0.54 / -0.2 / -0.32 vs PyTorch softmax) (Sec. V-B, Fig. 11) | `cimple_sw/quant.py`, `hf_attention.py`, `model.py`, `eval.py` | `uv run cimple-sw eval ...` (see below) | Pipeline runs end to end; only small `--limit` runs done so far. The main study (`results/run_main.sh`, see below) subsamples HellaSwag to 2000 docs |
| "33% reduction in latency" of activation-to-activation attention vs a non-split softmax at 400 MHz (Sec. IV-B, V-B) | `cimple_sw/latency.py` | `uv run cimple-sw latency` | **Not reproduced as-is:** 42.8% with the defaults (N=1024, d=64). Single documented baseline choices give 33.3-34.0% (`sm_reads=2`, `baseline_stream_norm`); the two-change combination `overlap_out` + `sm_reads=1` gives 32.7%. The paper doesn't say which baseline it used. |
| Mapping of encoder-only, decoder-only and encoder-decoder attention onto the dual-bank CIM core (Sec. IV-C/D/E) | `cimple_sw/mapping.py` | `uv run cimple-sw map [--n --d-model --heads --dec-tokens --e-mode]` | Bit-exact against an independent integer reference; cycle and utilisation statistics |
| Hardware constants (64x32 array, 2 banks, 8-cycle bit-serial MAC, 4-lane output mux, 128b write port). Clock: 400 MHz is the latency-study clock (Sec. V-B); the chip's operating range is 417-770 MHz (Table I) | `cimple_sw/hw.py` | - | Taken from the golden model and paper |
| Throughput metrics: OP convention, peak OP/cycle, TOPS at a given clock (Table I, note 4) | `cimple_sw/hw.py`, `latency.py`, `mapping.py` | `uv run cimple-sw latency --freq 770`, `uv run cimple-sw map --freq 770` | See [Performance metrics](#performance-metrics) |

Not covered here: energy and area (26.1 TOPS/W, 2.31 TOPS/mm2). Those come from
silicon and post-layout measurements, so software can't reproduce them.

## Performance metrics

- **OP convention** [P] (Table I, note 4): "one operation represents 1
  addition or multiplication", so 1 MAC = 2 OP.
- **Peak** [D]: 64 x 32 INT8 MACs per bit-serial pass, and a pass takes 8 MAC
  cycles, so 2 x 64 x 32 / 8 = **512 OP/cycle** (`hw.PEAK_OPS_PER_CYCLE`).
  That is 0.205 TOPS at 400 MHz, **0.214 TOPS at 417 MHz** and
  **0.394 TOPS at 770 MHz**. The paper reports no throughput TOPS figure, only TOPS/W and
  TOPS/mm2 (not covered here).
- **Delivered** [D]: the current RTL does not overlap the 8-phase output-mux
  walk with the next MAC, so a pass takes 8 + 8 = 16 cycles. That caps
  delivered throughput at 50% of peak. `--overlap-out` (an [A] shadow-register
  assumption) brings a pass back to 8 cycles.
- `uv run cimple-sw latency --no-sweep --freq 770` prints the peak in
  OP/cycle and TOPS, plus the delivered GOPS and core utilisation for the
  attention kernel. `uv run cimple-sw map --freq 770` prints delivered and peak
  GOPS and the utilisation per mapping. Both default to 400 MHz.

## Install and test

```bash
cd sw
unset VIRTUAL_ENV          # if another venv is active
uv sync                    # CPU torch from the pytorch-cpu index; cimple-rtl from ..
uv run pytest -q -m "not slow"   # ~250 tests, ~15 s
uv run pytest -q -m slow         # needs the TinyLlama checkpoint in the HF cache
```

## Softmax presets (`cimple_sw/config.py`)

| preset | what it is |
|---|---|
| `float` | The paper's baseline: the same W8A8 model with stock fp32 attention (eager). Q, K, V and the attention output stay fp32 |
| `float_int8attn` | Matched baseline **[A]**: fp32 softmax on the same per-(batch, head) INT8 Q/K/V, with the output requantised to INT8 in V's scale, exactly as `cimple` does. Only the softmax differs from `cimple` |
| `cimple` | Our reading of the paper's evaluated configuration: INT8 Q/K/V and scores, constant `z_max = 127`, full-precision (float64) exp table, exact reciprocal, INT8 output ("we chose full-precision tables to isolate the effect of the softmax approximation"). **[A]** The paper reads the reciprocal M from a full-precision LUT but does not give its index width or entry format; we model that table as an exact `1/sum`. No preset uses an fp32 reciprocal table indexed by 8 mantissa bits, so the full-vs-8b comparison is only exact for the exp table |
| `cimple_hw8` | Same, but uses the RTL's 8-bit UQ0.8 exp table and the 256-entry reciprocal LUT. Bit-exact with the golden model. |
| `cimple_hw8_rowmax` | Ablation: `cimple_hw8` with a per-row max instead of the constant 127 |

## Running the accuracy study

```bash
# quick smoke run (24 docs per task, ~1.5-1.7 min per preset on 16 threads; 5 min for all three)
uv run cimple-sw eval --presets float cimple cimple_hw8 \
    --tasks arc_challenge hellaswag openbookqa winogrande --limit 24 --out-dir results/limit24

# full run: one process per preset, then merge
uv run cimple-sw eval --presets float  --threads 16 --out-dir results/full
uv run cimple-sw eval --presets cimple --threads 16 --out-dir results/full
uv run cimple-sw eval --merge --out-dir results/full
```

### The main study as actually run (`results/run_main.sh`)

The headline numbers come from `results/run_main.sh`, **not** from the full
run above. It differs from that run in three ways:

| output dir | presets | tasks | docs | threads |
|---|---|---|---|---|
| `results/main3/` | `float`, `cimple`, `float_int8attn` (added afterwards, 8 threads) | ARC-C, OBQA, Winogrande | full splits | 11 |
| `results/hellaswag2000/` | `float`, `cimple` | HellaSwag | **first 2000 of 10042** (`--limit 2000`) | 11 |
| `results/ablation250/` | `float`, `float_int8attn`, `cimple_hw8`, `cimple_hw8_rowmax` | all four | 250 per task (`--limit 250`) | 4 |

Logs go to `results/logs/`. Each directory is merged on its own
(`cimple-sw eval --merge --out-dir results/<dir>`), so there are three
`summary.md` files and no single combined table. Each summary records its
`limit` in its header and the docs per task.

- **HellaSwag is a subset.** The paper presumably uses the full validation
  split. With acc_norm near 0.60, 2000 docs give a per-preset stderr of about
  1.1 points (sqrt(0.6 x 0.4 / 2000)), so the absolute HellaSwag numbers are
  not directly comparable to Fig. 11. The paired `cimple` - `float` delta has
  a smaller stderr, because both presets see the same 2000 docs. It can
  still be too coarse to confirm the paper's -0.54.
- **The ablations are smoke-scale.** At 250 docs per task the stderr is
  about 3 points. That is enough to show the `cimple_hw8` underflow (see
  Known limitations), and not enough to resolve sub-point deltas.

Each preset writes `<out-dir>/<preset>.json`, which holds the lm-eval results
plus a `per_task` record of docs, requests, wall time and requests/s.
`summary.md` is rebuilt from every JSON file in the directory. It shows
acc and acc_norm, the delta against `float`, the paper's values and the timing
per task. Measured at `--limit 24` on this machine (16 threads, CPU only),
extrapolated linearly to the full task sizes. These are upper bounds: a small
limit includes per-task setup and less efficient batching.

| preset | ARC-C (1172) | HellaSwag (10042) | OBQA (500) | Winogrande (1267) | total |
|---|---|---|---|---|---|
| float | 22 min | 3.0 h | 6 min | 14 min | ~3.7 h |
| cimple | 26 min | 3.5 h | 7 min | 15 min | ~4.3 h |
| cimple_hw8 | 26 min | 3.7 h | 7 min | 15 min | ~4.5 h |

HellaSwag (~40k loglikelihood requests) is ~80% of every run. Leaving it out
brings a preset under 50 min.

Useful flags: `--batch-size` (default 8), `--threads` (default: number of
physical cores), `--quant {w8a8,w8,dynamic,none}`, `--model <hf id>`,
`--num-fewshot`, `--calibrate` (switches the split_lut presets to static
per-layer score scales calibrated on built-in text; no effect on `float` /
`float_int8attn`; recorded as `calibrate` and `score_scale` in each JSON).

**What the delta measures.** `cimple` - `float` is *not* a softmax-only
comparison: besides the LUT softmax, `cimple` quantises Q, K, V and the
attention output to per-(batch, head) INT8, which `float` does not. On the
toy model in the tests, most of the `cimple` - `float` deviation comes from
that INT8 attention quantisation. Run `float_int8attn` alongside to isolate
the softmax; `summary.md` then adds `vs float_int8attn` columns. `merge` warns
when presets differ in model, quant, shots, limit, `--quantize-lm-head`,
batch size or `--calibrate`.

## Latency and mapping models

```bash
uv run cimple-sw latency                 # split vs baseline at N=1024, d=64, 400 MHz + sensitivity sweep
uv run cimple-sw latency --no-sweep --N 512 --d 64 --knob sm_reads=2
uv run cimple-sw latency --no-sweep --freq 770   # same at the top of the Table I clock range
uv run cimple-sw map --n 128 --d-model 256 --heads 4 --dec-tokens 16 [--freq 770]
```

## Assumptions

The paper leaves a lot unspecified. Every value it does not give is tagged
`[A]` in the code, with a one-line reason. Values from the paper's prose are
tagged `[P]`, values from a figure `[F]`, and derived values `[D]`. The module
docstrings are the authoritative list. The main assumptions:

- **Quantisation** (`quant.py`): W8A8 with per-output-channel weights and
  per-token dynamic activations, using fp32 fake-quant. lm_head and the
  embeddings stay floating point. The checkpoint is TinyLlama-1.1B-Chat-v1.0,
  evaluated 0-shot. The metric is guessed as acc_norm for ARC-C, HellaSwag and
  OBQA, and acc for Winogrande.
- **Softmax kernel** (`split_softmax.py`): Q, K and V are quantised to INT8
  per (batch, head). The dynamic score scale is the max |kept score| / 127
  (symmetric absmax).
  Scores are rounded in float, not by the RTL's integer scale/shift
  requantiser, so the kernel is bit-exact but the score requantisation is
  not.
- **HF hook** (`hf_attention.py`): right-padded lm-eval batches run each
  sequence on its real length, so pad tokens never enter the dynamic scales.
  This is not exact batch-size invariance: fp32 BLAS rounding in the linear
  layers (~1e-6) differs between batch shapes, and INT8 rounding in the
  kernel can amplify it to ~1e-3 nats per token, enough to flip a near-tie
  multiple-choice answer. Run the presets you compare at the same batch size.
  Static scales are calibrated per layer on fixed built-in text.
- **Latency** (`latency.py`): an in-order resource scheduler. The baseline
  softmax unit is fully pipelined at 4 values/cycle, which is generous to the
  baseline. `near_paper()` lists the knob settings that land near 33%.
- **Mapping** (`mapping.py`): bank writes overlap compute, the intermediate
  buffer holds one 32-column tile, KV-cache bandwidth is free, and only the
  attention block is modelled.

## Known limitations

- `cimple_hw8` (an 8-bit exp table with a constant `z_max`) underflows. Every
  exp entry of a row is 0 when that row's max score is far below 127, which
  hurts accuracy badly on TinyLlama. This is why the paper evaluates with
  full-precision tables. The absmax score scale makes it worse: a head whose
  negative tail exceeds its positive max can lose every row.
- The bit-exact `cimple_hw8` output rounding (golden model / RTL) is
  `floor(x - 1/2)` for negative values, a -1 LSB bias on non-tie negative
  outputs, although its comments say round-half-away.
- The per-(batch, head) dynamic INT8 scales (Q/K/V absmax and the score
  scale) are computed over the whole sequence, so in a causal prefill they
  see future tokens. No weight is ever placed on a future key, but row t's
  kept weights, hidden states and log-probs still change with tokens after t.
  lm-eval scores a continuation from one forward over `ctx + cont[:-1]`, so
  under the INT8-attention presets (`cimple*`, `float_int8attn`) every
  non-final continuation token's log-prob depends on the tokens after it.
  `float` has no such effect. On the toy model the effect is noise-level
  (~0.03 nats max, no systematic bias). For strict causality use static score
  scales and `quantize_qkv=False` (e.g.
  `PRESETS["cimple"].with_(score_scale="static", quantize_qkv=False)`).
- The `cimple` preset is ~1.1-1.2x slower than `float` on CPU, because of
  float64 integer emulation and a per-sequence loop for padded batches.

## Results

Results go in `sw/results/`. Each run directory (`results/<name>/`) has
per-preset JSON plus `summary.md`. The main study (`results/run_main.sh`)
writes its headline tables to `results/main3/summary.md` (ARC-C, OBQA,
Winogrande, full splits) and `results/hellaswag2000/summary.md` (HellaSwag,
2000-doc subset). Ablations go to `results/ablation250/summary.md`. A
`results/full/` directory exists only if the full run above is done. Smoke runs are
in `results/smoke/` (limit 16) and `results/limit24/` (limit 24). Their
sample sizes are too small to resolve the paper's sub-point deltas.
