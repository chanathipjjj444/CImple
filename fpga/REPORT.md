# CIMple on Versal VP1202: SystemVerilog re-implementation report

Paper: Ahn et al., *CIMple: Standard-cell SRAM-based CIM with LUT-based split
softmax for attention acceleration*, arXiv:2604.15944v2.
Design: `fpga/` (all new files). Your original RTL in `src/cimple_rtl` is
untouched. Contract (ISA, register map, semantics): `docs/ARCH.md`. Build and
flash: `vivado/README.md`.

## 1. TL;DR

- **It is built and verified in simulation.** The accelerator is a complete,
  synthesizable SystemVerilog design, from the CIM array up to an AXI4-Lite
  top. It is **bit-exact with your golden model** `cimple_model.py` on every
  check: raw MAC, projection, softmax row, a one-tile head (equal to
  `attention_head_cim` exactly), and a 128-token, 4-tile head.
- **Not done here: Vivado.** Vivado is not installed on this machine, so
  there is no PDI and no real timing report yet. The scripts are ready. Run
  `vivado/synth_ooc.tcl` first.
- **What you get from it.** A cycle-accurate, bit-exact platform for the
  paper's *dataflow*. It already answers several questions the paper leaves
  open (§4).
- **Biggest problems:**
  1. On an FPGA the 32 kb array is about 1% of what the VP1202's DSPs can do.
  2. The paper's 33% split-softmax latency gain largely disappears once the
     e "corner turn" is paid for.
  3. The end-to-end bottleneck is host data movement plus the 512 KiB global
     buffer, not compute (§6).

## 2. What was built

| layer | files | notes |
|---|---|---|
| CIM array | `rtl/cpl_bitcell, cpl_oai221, cpl_macro, cpl_partition, cpl_core` | Your structural bitcell + OAI221 macro. Additions: 32 partitions × 64 rows × 2 banks, a 128b write port (16 weights/cycle), an unsigned-activation mode for feeding e, and a pipelined adder tree (`PIPE_TREE=2`). |
| post-processing | `rtl/cpl_quant, cpl_lut, cpl_post` | ACC buffer (BRAM, 1024 × 4 × 32b), 4 requantisers (DSP), exp LUT RAM ×4, reciprocal ROM ×4 and 32 exp-sums. This is the split softmax of Fig. 7. |
| control | `rtl/cpl_engine, cpl_gbuf, cpl_imem, cpl_axil, cpl_top` | 5-opcode ISA (`WLOAD`, `MATVEC`, `LUTLOAD`, `NOP`, `END`), 256-instruction IMEM, 512 KiB global buffer (BRAM or URAM), AXI4-Lite slave, `irq_o` |
| host software | `host/cpl/isa, sim, compiler, driver` | Bit-exact ISA simulator. Compiler for projection, encoder head (IV-C), decoder step with KV cache (IV-D) and cross-attention (IV-E). Driver backends: sim, cocotb, `/dev/mem`/UIO on the A72, and JTAG via xsdb. |
| Vivado | `vivado/` | OOC synthesis; CIPS + SmartConnect block design → `.pdi`/`.xsa`; JTAG programming; xsdb smoke test. **Untested (no Vivado here).** |

About 2.9k lines of RTL.

## 3. Verification against the golden model

`tb/test_cpl_golden.py` (TOPLEVEL `cpl_top`) drives the RTL only through its
AXI4-Lite port. Every expected value comes from **your**
`src/cimple_rtl/model/cimple_model.py` primitives (`core_mac`, `quantise`,
`softmax_row`, `attention_head_cim`, …), not from my simulator or compiler.

| test | what is compared | result |
|---|---|---|
| raw matvec (RAW32) | 4 × 768 accumulators vs `gm.core_mac`, both banks, corners, partial reload | **bit-exact** |
| Q8 projection | 12 × 1920 outputs vs `gm.quantise`, K = 64 and 160 (ACC buffer), ties, saturation, extreme scales | **bit-exact** |
| softmax row (EXP → NORM) | e, partial, M/shift, score vs `gm.softmax_row`; sums 0, 10, 65280 | **bit-exact** |
| one-tile head | Q, K, V^T, e, out vs `gm.attention_head_cim` for (N, dm, d) = (32,64,32), (20,64,24), (32,64,16) | **bit-exact, no tolerance** |
| multi-tile head | N = 128, d = 64, 54 instructions, 1792 passes; reference built only from gm primitives | **bit-exact** |
| mutation check | rebuilt with `RND_FIX=1` (different rounding) | fails 924/1920, so the test is sensitive |

The rest of the verification:

- **Full regression:** `make -C fpga/tb regress` passes all 14 steps:
  - unit tests for every block, each under several parameter sets
  - the top level at 16 KiB, 64 KiB and 512 KiB global buffer, URAM and BRAM
  - a fuzzer of 250+ random legal and illegal programs, matched bit for bit
    against the ISA simulator
- **Host tests:** 128 pass.
- **Lint:** `verilator -Wall` is clean.
- **Synthesis structure:** `make synth-check` passes (URAM mapping, reset
  style).
- **Bug hunt:** an adversarial pass found **no functional divergence** in
  AXI corner cases, error paths, wraps, aliasing or reset mid-run.

## 4. What you get from this re-implementation (insights)

1. **The paper's dataflow works as described, and the gaps are now concrete
   decisions.**
   - Sign handling, the 128b write grouping, bank policy and LUT contents
     are all pinned down.
   - Feeding e through the CIM needs an **unsigned-activation mode**: e is
     0..255, but the array treats activations as signed.
   - Your golden model only gets this right because `softmax_row` uses
     `np.dot` instead of `core_mac`. With signed activations, every e > 127
     is wrong; the all-127-scores test catches this.
2. **The 16-cycle pass is the real ceiling, and it is achievable.**
   - Each pass is 8 bit-serial cycles plus 8 output-mux cycles, so the array
     can never exceed about 53% of its nominal 512 OP/cycle.
   - The RTL runs at **15.1 cycles/pass**, slightly better than the paper's
     16, because drain overlaps the next pass.
   - Projections reach **98% of that bound**.
3. **The 33% split-softmax claim depends on a hidden cost.**
   - The split softmax itself is free: EXP and NORM are fused into the
     QK^T / A'V passes and add 0.1 cycles/pass.
   - But the paper's encoder mapping (Q in the array, K streamed) produces
     e column-wise, while A'V needs it row-wise. That transpose ("corner
     turn") costs **78% of what a non-split softmax costs**.
   - Measured, the encoder attention ends up only **1-4% faster** than the
     non-split baseline.
   - Mapping K into the array instead (decoder-style) gives the same
     bit-exact output and is **25% faster per head** at N = 128-256.
   - With an ideal corner-turn buffer, the gain returns to about 40%,
     matching my earlier Python latency model.
4. **The constant z_max (127) wastes most of the 8-bit exp range.**
   - On real heads the per-row max score has a median of 53-65, not 127.
   - The largest e per row has a median of 15-40 out of 255, and 0.5-19% of
     e values are 0.
   - This, not the hardware, is the main accuracy loss (cosine 0.985-0.997
     vs float). It matches the TinyLlama ablation (`sw/REPORT.md` §3b), where
     8-bit tables lost 5-25 points with a constant max.
5. **Your golden model has a rounding bug, and the hardware copies it.**
   - `gm.quantise` rounds every negative value except exact .5 ties
     1 LSB too low; exact negative integers are included
     (`quantise(-4,1,1) = -3`).
   - Fixing it raises head cosine from 0.9955 to 0.9987 and removes a
     −1 LSB mean bias.
   - The RTL has `RND_FIX=1` ready. Flip it together with the golden model
     and the simulator's `rnd_fix` flag.

## 5. Still a problem

| # | problem | status |
|---|---|---|
| 1 | **No Vivado result yet.** Timing closure, real LUT/BRAM/URAM counts and the CIPS block-design Tcl are unverified. The CIPS `CONFIG` keys vary between Vivado versions. | Run `vivado/synth_ooc.tcl` (OOC, no board needed) first, then `build_versal.tcl`. |
| 2 | **Fmax.** The pre-fix estimate was 150-180 MHz, limited by the adder tree and the bank_sel/x_bits broadcast (fanout ~18k). The fixes (tree register at level 3, replicated controls, registered GBUF/ACC/quantiser paths) aim for 200-250 MHz. | Estimate only, from a yosys model; confirm with Vivado. |
| 3 | **BoostX specifics are unknown:** PMC reference clock, boot mode, JTAG reachability. Loading our PDI replaces MangoBoost's shell and drops the card's PCIe link, which can hang the host. | Marked in `bd_cips.tcl` as variables; check with the vendor and keep a recovery image. |
| 4 | **Host access only through AXI4-Lite, and only while idle.** The global buffer and IMEM refuse access during a run. | Design limitation of v1 (see §6). |
| 5 | **Rounding bug inherited from the golden model** (§4.5). | Your call: fix `gm.quantise`, then set `RND_FIX=1`. |
| 6 | **The spec doesn't define reset and power-up contents.** Banks and ACC have no reset; the simulator assumes zeros. | Documented as a spec gap; software must initialise. |

## 6. Bottlenecks (measured on the RTL, 200 MHz)

These were measured on the RTL before the timing fixes. The fixes add
latency but leave the pass rate unchanged at 15.1 cycles/pass, so the numbers
still hold.

| workload | cycles | cycles/pass | GOPS | % of 512 OP/cycle |
|---|---|---|---|---|
| projection matmul 128×256×256 | 66,910 | 16.3 | 50.2 | 49% (the ceiling is ~53%) |
| encoder head N=128, d=64 | 45,277 | 25.3 | 32.4 | 31% |
| BERT head N=128, d_model=768 | 183,217 | – | 45.8 | 45% |
| decoder step, context 512 | 7,626 | ~185 | 4 | 4% |

Where the time goes, worst first:

1. **Getting data in and out of the card (critical).**
   - One TinyLlama attention layer has 9.4 MB of INT8 weights, 18× the
     512 KiB global buffer. They are re-sent over the 32-bit AXI4-Lite port
     for every layer and every token.
   - Data transfer can't overlap compute, because the buffers reject host
     access during a run.
   - TinyLlama decode, attention only, across 22 layers: about **98 ms of
     compute per token**. Moving the data takes about **2.3 s/token** at A72
     speeds and about **44 s/token** with the current Python driver.
   - The fix is a DMA master to DDR over the NoC, plus a double-buffered
     global buffer moved to URAM.
2. **Weight loads never overlap compute (major).**
   - Loading one 32×64 weight tile takes 139 cycles, and runs strictly before
     the computation that uses it, even though the array has two banks.
   - That is **72% of decode cycles** and about 21% of each encoder A'V phase.
   - Overlapping them (load the idle bank while the active one computes)
     gives 1.12× on prefill and 1.38× on decode.
   - On top of that, decode should process the 8 query heads that share a
     key/value head together (TinyLlama uses grouped-query attention). That
     was measured at **5.1×**.
3. **Transposed output writes (major).** Writing outputs with a stride
   (`cstride ≠ 1`) goes 1 byte/cycle, so a pass takes 33 cycles instead of
   15. That is 32% of an N = 256 encoder head. A write-combining transposer,
   or the K-in-array mapping, removes it.
4. **The pass structure itself.** The 8-phase output mux caps sustained
   throughput at ~53% of peak. A shadow accumulator would let drain fully
   overlap the next MAC.
5. **Small fixed costs.** 31 cycles per MATVEC (7 issue + 24 pipeline fill),
   a 256-instruction IMEM and a 1024-entry ACC buffer.
   - A TinyLlama layer is about 10k instructions, so about 42 IMEM reloads.

**FPGA reality check.** The 32 kb array peaks at 0.10 TOPS and sustains
0.03-0.05 TOPS at 200 MHz. The VP1202's 3,984 DSP58s give at least about
4 INT8 TOPS, so CIMple uses about **1% of the device's compute** while taking
about 9% of its LUTs. The flip-flop "bitcells" are faithful to the paper but
are not an efficient FPGA structure.

On the BoostX, the real near-memory win would come from:
- putting the engine next to DDR/HBM with DMA;
- reusing the same ISA and dataflow over a DSP58 array or AIE-ML.

## 7. Recommended next steps

1. **Vivado.** Run `vivado -mode batch -source fpga/vivado/synth_ooc.tcl` and
   look at WNS and utilisation. Then confirm the VP1202 part string and the
   BoostX CIPS settings, and run `build_versal.tcl`.
2. **Decide on rounding.** Fix `gm.quantise`, then set `RND_FIX=1`; all tests
   are parameterised for it.
3. **Cheap performance wins (compiler only):**
   - use the K-in-array mapping for encoder QK^T (−25% cycles)
   - batch the query heads that share a KV head in decode (5.1×)
4. **RTL v2:**
   - overlap weight loads with compute (ping-pong banks)
   - corner-turn buffer for transposed writes
   - an AXI4 DMA master to DDR
   - a double-buffered global buffer in URAM
5. **Research angle:** replace the constant z_max with a coarse per-tile max.
   This is cheap in hardware; it recovered TinyLlama accuracy to within
   1-2 points in the Python study.

## 8. How to run

```bash
cd /home/chanathip/playground/CImple && unset VIRTUAL_ENV
uv run make -C fpga/tb regress                       # full RTL regression (14 steps)
uv run make -C fpga/tb MODULE=test_cpl_golden TOPLEVEL=cpl_top PARAMS="-GGBUF_BYTES=65536"
(cd sw && uv run pytest ../fpga/host/tests -q)       # host ISA sim / compiler / driver
(cd sw && uv run python ../fpga/host/run_demo.py)    # encoder head on the ISA simulator
# on a Vivado machine:
vivado -mode batch -source fpga/vivado/synth_ooc.tcl     # OOC synthesis + timing, VP1202
vivado -mode batch -source fpga/vivado/build_versal.tcl  # CIPS BD -> .pdi / .xsa
```

## 9. Resource estimate (final RTL, yosys, not Vivado)

How it was produced:
- converted with sv2v, then synthesised by yowasp-yosys 0.70 using
  `synth_xilinx -family xcup -flatten` on `cpl_top`
- 64 KiB global buffer, all other parameters at their defaults
- the target family is UltraScale+, the closest yosys has to Versal

| LUT2-6 | FF | CARRY4 | DSP48E2 | RAMB36E2 | RAM32M16 (LUTRAM) | SRL16E |
|---|---|---|---|---|---|---|
| 56,666 | 44,218 (0 with async reset) | 2,190 | 11 | 24 | 1,064 | 387 |

- **Memory:** the 32,768 bitcells are flip-flops; the 24 RAMB36 are the
  64 KiB global buffer (16), IMEM (4) and ACC buffer (4).
- **Default 512 KiB buffer:** expect about 128 RAMB36, or URAM288 with
  `GBUF_RAM_STYLE="ultra"`.
- **Fit:** the VP1202 has >900k LUTs, so the whole accelerator is roughly
  6-8% of the device.
- **Caveat:** Vivado's mapping onto Versal (DSP58, 8-bit carry chains,
  LUTRAM) will differ. Use `vivado/synth_ooc.tcl` for the real numbers.
