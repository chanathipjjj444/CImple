# CIMple — bit-level spec reconstructed from the paper

Source: Ahn, Tao, Gomony, Geilen, Corporaal, *"CIMple: Standard-cell SRAM-based
CIM with LUT-based split softmax for attention acceleration"*, arXiv:2604.15944v2
(TU Eindhoven, Electronic Systems).

The paper gives an architecture figure with width annotations but no RTL and no
timing diagram. Everything below is a reconstruction. Each number is tagged:

- **[P]** stated in the paper's prose
- **[F]** read off the annotations in Fig. 4
- **[D]** derived — the only value consistent with [P] and [F]
- **[A]** an assumption I made where the paper is silent. These are the places
  where your implementation may legitimately diverge from theirs.

---

## 1. Array geometry

| Quantity | Value | Evidence |
|---|---|---|
| Total CIM SRAM | 32 kb | [P] "a 32kb CIM core" |
| Partitions | 32 | [P] "In total there are 32 partitions of this CIM core" |
| SRAM blocks per partition | 2 × 512 b | [P] "two SRAM blocks of 512 bits" |
| Nibble split | top half = MSB 4 b, bottom = LSB 4 b | [P] |
| Macro | 16 bitcells + 8 OAIs | [P] "a macro is created of 16 bitcells and 8 OAIs" |
| Input width | 64 b | [P] "The input bit width is 64b" |
| Write data width | 128 b | [P] |
| Bitcell | 8T, from Mori et al. ISSCC'23 (ref [7]) | [P] |

### The arithmetic that makes these consistent

2 banks × 32 partitions × 64 rows × 8 b = **32 768 b = 32 kb**. ✔ [D]

Walk it from the macro up:

- One macro = 16 bitcells = **two INT8 weights**, one in each bank. [D]
- Its 8 OAIs produce 8 product bits from **one** activation bit: 4 bits of
  MSB-nibble product, 4 bits of LSB-nibble product. One activation bit per
  macro. [D]
- 64 macros per partition ⟺ the 64 b input width. ✔ [D]
- 64 macros × 2 weights × 8 b = 1024 b = the two 512 b blocks. ✔ [D]
- Fig. 4's `4b x 2048` on the adder-tree input is **32 partitions × 64 rows =
  2048 nibbles**, not a per-partition number. [D] This is the annotation most
  likely to mislead you on a first read.

So the CIM core is a **64-input × 32-output INT8 matrix-vector engine**, with
two resident weight sets (banks) of which one is active. 2048 MACs per 8-cycle
pass = 256 MAC/cycle = 512 OP/cycle.

> Sanity check against the paper's numbers: 512 OP/cycle × 770 MHz ≈ 0.39 TOPS
> peak. The headline 26.1 TOPS/W is an *efficiency*, not a throughput, and is
> quoted post-synthesis with 87.5 % activation sparsity (Fig. 8), so the two
> are not in conflict — but do not expect 26 TOPS out of this array.

---

## 2. Dataflow and the width chain

Weights are **8-bit parallel** (the paper's contribution 1: "8-bit parallel
weight feeding") — both nibbles of a weight are available every cycle.
Activations are **bit-serial over 8 cycles** [P] "It accumulates partial
products over 8 cycles to get the complete 8x8 bit results".

Every width in Fig. 4 falls out of that:

| Node | Width | Derivation |
|---|---|---|
| Nibble product | 4 b | 1 b activation AND 4 b weight nibble |
| Adder tree out | **10 b** [F] | 64 × 4-bit nibbles: signed MSB tree ∈ [−512, 448]; unsigned LSB tree ∈ [0, 960] |
| After `<<4` + add | **15 b** [F] | (10 b signed ≪ 4) + 10 b unsigned ∈ [−8192, 8128] |
| After `<<1 & ACC` ×8 | **23 b** [F] | 15 b + 7 bits of growth + sign headroom |
| CIM OUT | 23 b × 4 lanes [F] | 32 partitions muxed 8:1 by a 3 b counter [F] |
| Intermediate ACC / buffer | 32 b × 4 [F] | |
| Quantiser | 32 b → 8 b, scale 4 × 32 b [F] | |
| LUT in / out | 4 × 8 b in, 128 b out [F] | |

Every single one matches. That agreement across five independent nodes is the
main reason to trust this reconstruction.

### Sign handling — **[A], and the most likely place to diverge**

The paper never says how INT8 signs are handled. The scheme in `rtl/` is:

- **MSB nibble signed, LSB nibble unsigned.** Then `w = msb·16 + lsb` *is*
  the two's-complement INT8 value, exactly, with no correction term. This is
  why the MSB adder tree is signed and the LSB tree is not, and it is the only
  split I found that makes the 10 b / 15 b widths work out as annotated.
- **Activations MSB-first**, cycle 0 carrying the sign bit:
  ```
  x = -x7·2^7 + x6·2^6 + … + x0
  acc ← −P7 ;  acc ← 2·acc + P6 ;  … ;  acc ← 2·acc + P0
  ```
  i.e. the first `<<1 & ACC` cycle subtracts. LSB-first with a final negate is
  equally valid and gives identical results; it costs a wider accumulator.

`model/cimple_model.py::partition_mac` asserts the result equals
`np.dot(w, a)` for every case, including all four signed corners.

---

## 3. LUT-based split softmax (paper §IV-B, Fig. 7)

The contribution is **two independent simplifications**, and it is worth
keeping them separate in your head because they have very different costs:

**(a) Constant max instead of running max.** Safe softmax subtracts
`z_max` per row, which forces a full pass over the row before any exponential
can be computed. CIMple replaces it with `z_quant_max` — the largest
*representable* quantised value (127) — which is a constant. [P] "The maximum
value of the input is replaced by the maximum possible quantization value".
Cost: the exponentials are smaller than they need to be, so the 8-bit LUT
output range is used less efficiently. Benefit: no blocking pass.

**(b) Split numerator and denominator.** `exp(ż_i)` streams straight into the
`A'V` MAC as it is produced, while the denominator accumulates in parallel.
Only at the end of the row is the reciprocal applied. [P] 33 % latency
reduction versus a non-split LUT softmax, at 400 MHz, head dim 64, 1024 tokens.

### Table definitions — **[A]** (the paper gives only "256 × 8b" and "M")

```
EXP_LUT[i]   = round(255 · exp((int8(i) − 127) · score_scale))     i ∈ [0,255]
RECIP_LUT[m] = round(2^15 / (256 + m))                             m ∈ [0,255]
```

The exp table absorbs three operations so the hardware does none of them: the
dequantisation of the INT8 score, the `− z_quant_max` subtraction, and the
requantisation to UQ0.8. Index it with the raw 8-bit score pattern.

The reciprocal is indexed by the 8 bits **below the leading 1** of the exp-sum
`S`, so it inverts a normalised mantissa `1.m`:

```
S      = (256 + m) · 2^(e−8)          e = position of the MSB of S
1/S   ≈ RECIP_LUT[m] · 2^(−15) · 2^(8−e)
score  = sat8( (Σ e_j·V_j · M) >> (15 − 8 + e) )
```

The `+8` from the UQ0.8 exponentials cancels the `−8` in the mantissa
normalisation, which is why the shift is `RECIP_FRAC − 8 + e`.

### Known weakness of the 8-bit table

At `score_scale = 0.05`, only **125 of 256** exp entries are non-zero — the
rest underflow. That is a direct consequence of simplification (a): because
the offset is the constant 127 rather than the row's actual max, a row whose
scores all sit well below 127 lands entirely in the underflow region and the
softmax degenerates. The paper sidesteps this by evaluating with
*full-precision* tables [P] "we chose full-precision tables to isolate the
effect of the softmax approximation". **This is a real open question in their
design, and a legitimate contribution angle for you** — see
`docs/02-roadmap.md` §"Where there is actual research here".

`model/gen_luts.py` prints the non-zero count so you notice when you have
chosen a bad scale.

---

## 4. What is and is not reproduced

| Paper | This repo |
|---|---|
| 8T SRAM bitcell as a custom standard cell, with hand-drawn LIB/LEF | Plain flops in `cim_macro.sv` |
| OAI221 as 1 b multiplier **and** bank selector | A plain AND |
| ST 28 nm FD-SOI, Genus / Virtuoso / Innovus, digital-on-top | Nothing |
| 26.1 TOPS/W, 2.31 TOPS/mm² | **Not reproducible here, by construction** |
| Dataflow, widths, bit-serial schedule, nibble split, mux/counter | Reproduced exactly |
| Split-softmax LUT scheme and its accuracy behaviour | Reproduced, parameterised |

The divergence is confined to **one file** — `rtl/cim_macro.sv`. Everything
above it is the real architecture. If you later want the circuit-level story
back (OpenROAD + sky130/gf180, or a real PDK through CMKL), you replace that
one module and the rest of the tree is untouched. That isolation is the point
of the structure, not an accident.

---

## 5. Gaps the paper leaves open

Flag these if you write this up; they are also the easiest places to get a
different answer than the authors did and not know why.

1. **Sign handling** — not specified anywhere. See §2.
2. **Sparsity exploitation.** Fig. 8 reports TOPS/W at 50 / 75 / 87.5 %
   activation sparsity, but no gating mechanism is described. Zero activation
   bits naturally produce zero OAI output, so the saving may be pure switching
   activity rather than an architectural skip. If it is just switching
   activity, it cannot be reproduced in RTL simulation at all — only in power
   analysis with real toggle data.
3. **Bank usage policy.** Two banks exist and only one can be active, but the
   paper does not say whether the second bank is double-buffering the next
   weight tile, holding K vs V, or something else. The mapping sections hint
   at K^T and V both living in the CIM for decoder-only, which suggests
   double-buffering across the two blocks.
4. **Exp-sum accumulator width.** Never stated. 24 b here, which is safe for
   rows up to ~65 k tokens at 8 b exponentials.
5. **Where the `A'V` partial lives.** The 32 b × 4 intermediate buffer is
   shown, but the depth is not given — it bounds the maximum head dimension
   you can process without spilling to the global buffer.
6. **The 128 b write path vs the 64 b read path.** Write is 128 b, i.e. 16
   INT8 weights per cycle, so loading one partition's 64 weights takes 4
   cycles and the whole core 128 cycles. The RTL here writes one weight per
   cycle for simplicity — widen `cim_partition`'s write port when you care
   about weight-load latency.
