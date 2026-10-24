# CIMple-FPGA (`cpl_*`) — architecture and ISA contract

The CIMple self-attention accelerator (Ahn et al., arXiv:2604.15944v2, Fig. 4 /
Fig. 7, Sec. IV) as a synthesizable SystemVerilog design for the AMD Versal
Premium **VP1202** (`xcvp1202-vsva2785-2MP-e-S` by default), driven by a host
over **AXI4-Lite**.

This document is the **contract**: the RTL (`fpga/rtl`), the bit-exact Python
ISA simulator (`fpga/host/cpl/sim.py`) and the compiler (`fpga/host/cpl/compiler.py`)
are implemented independently from it and cross-checked against each other.
If the RTL and this document disagree, the RTL is wrong.

Tags: **[P]** paper prose, **[F]** paper figure, **[D]** derived, **[A]** our
assumption (the paper is silent).

The user's original RTL in `src/cimple_rtl/rtl` is **not modified**; `fpga/rtl`
starts from copies of its verified datapath (`bitcell`, `oai221`,
`cim_macro`, `cim_partition`, `cim_core`, `quant_unit`) at commit `d195ccd`,
renamed `cpl_*` so both trees can coexist, and extends them as listed in §2.

---

## 1. Block diagram

```
           AXI4-Lite (32b data, 21b address)
                    |
               cpl_axil  -- regs, IMEM window, GBUF window
                    |
   +----------------+------------------------------------------------+
   |  cpl_engine (fetch / decode / execute, one instruction at a time) |
   |                                                                   |
   |   IMEM (256 x 256b)        GBUF (GBUF_BYTES, 128b lines, 2 ports)  |
   |        |                     | rd (acts, weights, LUT)  ^ wr (outs)|
   |        v                     v                          |         |
   |    decode ---> XIN reg (64 x 8b) --> cpl_core (32 part x 64 rows   |
   |                 bit-serial MSB first     x 2 banks, 128b write)   |
   |                                            | 23b x 4 lanes, 8 phases
   |                                            v                      |
   |                 cpl_post: ACC buffer (32b x 4) -> quant x4 ->      |
   |                           exp LUT x4 / recip LUT x4 / exp-sums     |
   |                                            | bytes / words        |
   |                                            v                      |
   |                 output writer (strided byte writes into GBUF) ----+
   +-------------------------------------------------------------------+
```

## 2. Datapath (what changed vs `src/cimple_rtl`)

| item | value | note |
|---|---|---|
| array | 32 partitions x 64 rows x 2 banks x INT8 = 32 kb [P] | flip-flop bitcells (`cpl_bitcell`), structural OAI221 macro, as in the original |
| MAC | bit-serial, 8 cycles, MSB first, sign cycle subtracts [P][D] | unchanged |
| **unsigned activations** | `act_unsigned=1`: the MSB cycle **adds** instead of subtracting, so activations are UQ8 (0..255) [A] | needed to feed exp values e (UQ0.8) into A'V without the `e>>1` precision loss (see sw/REPORT.md §6.2) |
| **write port** | 128b = 16 INT8 weights per cycle [P]: one partition, row group `wgrp` (rows `16*wgrp .. 16*wgrp+15`), byte i -> row `16*wgrp+i` | the original writes 8b/cycle |
| **adder-tree pipeline** | parameter `PIPE_TREE` (0..2, default 2): 1 = register after the two adder trees + `<<4` combine; 2 = in addition a register after adder level 3; the shift-accumulator update happens `PIPE_TREE` cycles after the bit is presented | timing on Versal at 200 MHz [A] (default raised from 1 to 2 by the timing review, item T1; results unchanged, MAC latency +1) |
| output select | 32 -> 4 lanes x 8 phases: lane `l`, phase `k` = partition `8l+k` [F] | phase is driven by the engine (`out_phase_i`), no free-running counter |
| ACC buffer | `ACC_DEPTH` entries (default 1024) x 4 lanes x 32b, RMW, wraps mod 2^32 [A] | BRAM |
| quantiser | `sat8(rnd(v * scale >> shift))` with **exactly** the golden-model rounding (`gm.quantise`: `+2^(s-1)` for v*scale >= 0, `-2^(s-1)` otherwise, then arithmetic shift) | bit-exact with `cimple_model.quantise` incl. its known -1 LSB bias on negative non-ties (kept for bit-exactness; see §9) |
| exp LUT | 256 x 8b **RAM**, 4 replicas (one per lane), loaded by `LUTLOAD` [A] | contents = `gm.build_exp_lut(score_scale)`, depends on the score scale, so not a ROM |
| recip LUT | 256 x 8b ROM, `round(2^15/(256+m))` = `gm.build_recip_lut()`, 4 replicas, initialised by a constant function (no hex file) | |
| exp-sums | 32 x 24b registers `SUMS[0..31]`, wrap mod 2^24 [A] | the split-softmax denominator [P] |
| GBUF | `GBUF_BYTES` (default 512 KiB) of 128b lines, true dual port | the paper's "global buffer" (16 kb in silicon [P]; larger here) |
| IMEM | 256 instructions x 256b | |
| clock | single domain, target 200 MHz [A] | |

## 3. Host interface: AXI4-Lite slave

32-bit data, `ADDR_W = 21` (2 MiB window), byte strobes honoured, OKAY/SLVERR responses.
All accesses must be 32-bit aligned (unaligned -> SLVERR).

| address | name | access | meaning |
|---|---|---|---|
| `0x000000` | `ID` | RO | `0x43504C31` ("CPL1") |
| `0x000004` | `CTRL` | WO | bit0 `START` (self-clearing; ignored while busy), bit1 `CLEAR` (clears `done`/`error`/err code) |
| `0x000008` | `STATUS` | RO | bit0 `busy`, bit1 `done` (sticky), bit2 `error` (sticky), bits[15:8] `err_code`, bits[31:24] reserved 0 |
| `0x00000C` | `PC_START` | RW | bits[7:0] first instruction index |
| `0x000010` | `PC` | RO | index of the instruction being executed / that ended the run |
| `0x000014` | `CYCLES` | RO | clock cycles from START to done/error of the last run |
| `0x000018` | `INSTRET` | RO | instructions retired in the last run (END counts) |
| `0x00001C` | `PASSES` | RO | MAC passes (activation vectors) executed in the last run |
| `0x000020` | `CFG` | RO | bits[15:0] `GBUF_BYTES/1024`, bits[31:16] `IMEM_DEPTH` |
| `0x010000 + 32*i + 4*w` | `IMEM[i].word[w]` | RW | instruction i (0..255), 32-bit word w (0..7) |
| `0x100000 + a` | `GBUF[a..a+3]` | RW | little-endian 32-bit word of GBUF bytes a..a+3 (a multiple of 4, `a < GBUF_BYTES`) |

* IMEM and GBUF accesses while `busy` return **SLVERR** (writes are dropped,
  reads return 0). Register reads always work.
* Unmapped addresses: reads return 0 with SLVERR; writes are dropped with SLVERR.
* `START` clears `done`, `error`, `err_code`, `CYCLES`, `INSTRET`, `PASSES`, clears
  the 32 exp-sums and sets `PC = PC_START`. ACC buffer, banks, exp LUT and GBUF are **not** cleared.

## 4. Instruction format (256 bits = 8 little-endian 32-bit words)

`w0`:

| bits | field | meaning |
|---|---|---|
| [3:0] | `opcode` | 0 `END`, 1 `WLOAD`, 2 `MATVEC`, 3 `LUTLOAD`, 4 `NOP`; others -> error 1 |
| [4] | `bank` | bank written (WLOAD) / computed on (MATVEC) |
| [6:5] | `acc_mode` | 0 `NONE`, 1 `FIRST`, 2 `ADD`, 3 reserved (-> error 1) |
| [9:7] | `out_mode` | 0 `NONE`, 1 `Q8`, 2 `EXP`, 3 `NORM`, 4 `RAW32`, 5-7 reserved (-> error 1) |
| [10] | `act_unsigned` | activations are UQ8 instead of INT8 |
| [11] | `sum_by_t` | exp-sum / norm index = `t mod 32` (else = column `p`) |
| [12] | `sum_clr` | clear all 32 exp-sums before this instruction |
| [13] | `causal` | causal mask in `EXP` mode |
| [15:14] | reserved | must be 0 |
| [31:16] | `ntok` | number of activation vectors t = 0..ntok-1 (0 allowed: no passes) |

| word | MATVEC | WLOAD | LUTLOAD |
|---|---|---|---|
| `w1` | `aaddr` (byte address of x_0) | `gaddr` | `gaddr` |
| `w2` | `astride` (bytes between x_t) | `pstride` (bytes between partitions) | - |
| `w3` | `oaddr` | [4:0] `p0`, [13:8] `np` (1..32) | - |
| `w4` | [15:0] `tstride`, [31:16] `cstride` (bytes) | - | - |
| `w5` | [15:0] `scale`, [20:16] `shift`, [29:24] `ncols` (0..32) | - | - |
| `w6` | [15:0] `acc_base` | - | - |
| `w7` | [15:0] `qbase`, [31:16] `kbase` | - | - |

Unused words/fields are ignored (should be 0).

## 5. Instruction semantics (normative)

Notation: `G[a]` = GBUF byte a. `s8(b)` = byte as signed INT8, `u8(b)` = as unsigned.
`W[b][p][r]` = bank b, partition p, row r (INT8). `A[e][l]` = ACC buffer entry e, lane l (32b).
`S[i]` = exp-sum i (24b unsigned). `quantise(v, scale, shift)` = golden model `gm.quantise`
on exact integers, saturating to [-128, 127]. All address arithmetic is on unbounded
integers; any byte address touched outside `[0, GBUF_BYTES)` is error 3.

**END**: stop; `done=1`. **NOP**: nothing.

**WLOAD** (bank, gaddr, pstride, p0, np): require `gaddr % 16 == 0`, `pstride % 16 == 0` (else error 2),
`1 <= np` and `p0 + np <= 32` (else error 4). For i in 0..np-1, r in 0..63:
`W[bank][p0+i][r] = s8(G[gaddr + i*pstride + r])`. Other partitions / the other bank unchanged.

**LUTLOAD** (gaddr): require `gaddr % 16 == 0` (error 2). `EXPLUT[k] = G[gaddr + k]`, k = 0..255 (all 4 replicas).

**MATVEC**: require `aaddr % 16 == 0` and `astride % 16 == 0` (error 2); `ncols <= 32` (else error 4);
the reserved encodings above -> error 1. If `sum_clr`: `S[i] = 0` for all i. Then for t = 0..ntok-1, in order:

1. `x[r] = (act_unsigned ? u8 : s8)(G[aaddr + t*astride + r])`, r = 0..63.
2. `acc[p] = sum_r W[bank][p][r] * x[r]` (exact; fits 23b signed), p = 0..31.
3. For every p = 0..31 (all columns, regardless of `ncols`): `e = acc_base + 8*t + (p mod 8)`,
   `l = p div 8`, entry index taken mod `ACC_DEPTH`;
   * `NONE`:  `v[p] = acc[p]` (sign-extended to 32b)
   * `FIRST`: `A[e][l] = acc[p]`; `v[p] = A[e][l]`
   * `ADD`:   `A[e][l] = wrap32(A[e][l] + acc[p])`; `v[p] = A[e][l]` (v as signed 32b)
4. For p = 0..ncols-1, with `o = oaddr + t*tstride + p*cstride`:
   * `NONE`: nothing.
   * `Q8`:   `G[o] = quantise(v[p], scale, shift)` (as a byte).
   * `EXP`:  `z = quantise(v[p], scale, shift)`;
     `(q, k) = sum_by_t ? (qbase + t, kbase + p) : (qbase + p, kbase + t)`;
     `ev = (causal and k > q) ? 0 : EXPLUT[u8(z)]`; `G[o] = ev`;
     `i = sum_by_t ? (t mod 32) : p`; `S[i] = (S[i] + ev) mod 2^24`.
   * `NORM`: `i = sum_by_t ? (t mod 32) : p`; `s = S[i]`;
     `msb = (s == 0) ? 0 : floor(log2 s)`; `m = mantissa8(s)` (= gm.mantissa8: `(s >> (msb-8)) & 0xFF` if msb >= 8 else `(s << (8-msb)) & 0xFF`);
     `G[o] = quantise(v[p], RECIP[m], 7 + msb)`.
   * `RAW32`: `G[o..o+3] = v[p]` little-endian (two's complement).
   Within one t, writes happen in increasing p; later writes win on overlap.
   Writes of step t are visible to the activation read of step t+1 (the engine must not
   prefetch x_{t+1} across an output write that could alias it — or must handle the hazard).

Errors stop execution immediately with `error=1`, `err_code` set, `PC` = faulting instruction,
`done=0`. Error codes: 1 bad opcode/reserved encoding, 2 misalignment, 3 GBUF address out of
range, 4 bad partition range / ncols > 32. All errors are detected **before** the instruction
changes any state, except error 3 inside MATVEC, which may be raised mid-instruction
(the state changes made so far by that instruction are unspecified).

Fetching past instruction 255 -> error 1.

## 6. Timing (informative; the cycle counts are not part of the contract)

* MATVEC per t: 4 GBUF line reads for x_t (may be prefetched), 8 MAC cycles (+`PIPE_TREE`), 8 drain cycles (4 lanes each) through `cpl_post`, then the output writer.
  Contiguous byte outputs (`cstride == 1`) are written as whole/partial 128b lines with byte
  enables; other strides cost one cycle per byte.
* WLOAD: 4 cycles per partition (one 128b line per cycle) + read latency.
* The paper's 16-cycle pass (8 MAC + 8 drain) is the lower bound per t.

## 7. Software stack (`fpga/host/cpl`)

* `isa.py` — field encode/decode, `Instr` dataclass, `assemble()/disassemble()`.
* `sim.py` — bit-exact ISA simulator of §5 (numpy), the reference for RTL tests.
* `compiler.py` — memory planner + program builders for: tiled INT8 matmul (weight projection),
  encoder-only attention head (Sec. IV-C: Q and V^T in CIM, K streamed), decoder-only step with
  KV cache (Sec. IV-D: K and V^T in CIM, q streamed), encoder-decoder cross attention (IV-E).
* `driver.py` — `Device` over an MMIO backend: `SimBackend` (ISA sim), cocotb backend (tb),
  `DevMemBackend` (Linux /dev/mem or UIO on the Versal APU), `XsdbBackend` (JTAG via xsdb).

## 8. Files

```
fpga/
  docs/ARCH.md          this contract
  rtl/                  cpl_pkg.sv cpl_bitcell.sv cpl_oai221.sv cpl_macro.sv cpl_partition.sv
                        cpl_core.sv cpl_quant.sv cpl_lut.sv cpl_post.sv cpl_gbuf.sv cpl_imem.sv
                        cpl_engine.sv cpl_axil.sv cpl_top.sv
  tb/                   cocotb + Verilator tests (unit + top-level via an AXI4-Lite BFM)
  host/cpl/             isa.py sim.py compiler.py driver.py
  vivado/               synth_ooc.tcl  build_versal.tcl  bd_cips.tcl  constraints  program.tcl
```

## 9. Known deviations / open points

* Rounding: `quantise` keeps the golden model's `floor(x - 1/2)` behaviour for negative
  values (a -1 LSB bias on negative non-ties) so the design stays bit-exact with
  `cimple_model.py`. A parameter `RND_FIX` (default 0) in `cpl_quant` switches to true
  round-half-away; the ISA simulator takes the same flag.
* Bitcells are flip-flops: on an FPGA "compute-in-memory" is near-memory compute; energy/area
  numbers of the paper do not transfer.
