# Roadmap: open-source sim → DSim → Vivado

## The honest framing first

CIMple is an **ASIC paper**. Its headline results (26.1 TOPS/W, 2.31 TOPS/mm²)
come from a custom 8T bitcell standard-cellised in ST 28 nm FD-SOI, synthesised
in Genus and placed in Innovus. None of that survives a move to FPGA. An FPGA
"CIM" is a contradiction in terms — the array becomes BRAM or LUTRAM and the
in-memory-compute saving disappears entirely.

So be clear with yourself (and with Dr. Sangpetch) about what an FPGA
implementation of this is *for*:

- ✅ **Architectural validation** — the bit-serial schedule, the nibble split,
  the split-softmax pipeline, the mapping of encoder/decoder attention onto a
  64×32 array, cycle counts, accuracy of the INT8 LUT softmax. All of this is
  real, publishable work and all of it is FPGA-verifiable.
- ✅ **A platform for your own contribution.** The interesting questions the
  paper leaves open (§5 of `01-spec-teardown.md`) are architectural, not
  circuit-level.
- ❌ **Energy or area claims.** Do not put an FPGA power number next to their
  TOPS/W. It is not the same measurement and a reviewer will say so.
- ❌ **Reproducing the paper.** You cannot, without a 28 nm PDK. Say
  "re-implementation of the architecture" and you are on solid ground.

If you *do* want the circuit story, the open path is OpenROAD + sky130 or
gf180 with an OpenRAM-generated or hand-drawn bitcell — but that is a separate
project, and at sky130's 130 nm the efficiency numbers will look nothing like
28 nm. Treat it as a later, optional branch, not the main line.

---

## Milestones

### M0 — reproduce the datapath (this repo) ✅
CIM core + quantiser + split softmax, bit-exact against a numpy golden model,
under Verilator + cocotb. Done; `make -C tb regress`. Measured LUT-softmax error vs float: mean 1.43 LSB,
max 6.03 LSB over random rows at `score_scale = 0.05`.

### M1 — close the spec gaps
Work through `01-spec-teardown.md` §5. Each gap is a decision you have to
make explicitly and record. In particular pick and justify the sign scheme and
the bank policy, because everything downstream depends on them.

Deliverable: a one-page design-decision log. Your supervisor will ask.

### M2 — the mapping FSM
This is the part of the paper that makes it an *accelerator* rather than a
macro: weight projection → QK^T → softmax → A'V, with tiling, for encoder-only
and decoder-only. The paper's §IV-C/D/E describe it only in prose.

Build it as a separate `cimple_seq.sv` sequencer driving the M0 datapath.
Verify against a numpy attention reference with the same quantisation — one
layer of a small BERT or TinyLlama, not a toy.

This is also where you will discover whether 64×32 is actually a sensible
array shape for the head dimensions you care about. It may not be.

### M3 — second simulator: DSim
See "Tooling path" below. Run the same cocotb suite under DSim. The point is
not speed; it is that two independent simulators disagreeing finds real bugs,
especially around X-propagation and signed arithmetic, which Verilator is
famously permissive about.

### M4 — FPGA bring-up
Vivado xsim first (third simulator, free, and the one whose semantics match
what will actually be synthesised), then synthesis, then a board.

Expect the array to be the bottleneck: 32 partitions × 64 macros × 2 banks ×
8 b = 32 kb of storage, but with 64 independent read ports per partition. A
naive mapping will not infer BRAM — it will burn ~32 k flops plus a very wide
combinational reduction. Plan for this:
- Flop-based array: fine on a mid-size part, terrible timing on a small one.
- BRAM-based: you must restructure the read so each BRAM serves a column, which
  changes the schedule. This is real design work, not a port.
- Start with a **reduced configuration** (`N_PART=8`, `N_ROWS=16`) to get the
  flow working end to end, then scale. The RTL is parameterised for this.

### M5 — Vivado Enterprise / Versal
Only once M4 closes on a smaller part. Per your earlier licensing work this
needs PRO minimum, Enterprise preferred via the University Program. Do not let
the licence block M0–M4; none of them need it.

---

## Tooling path

| Stage | Tool | Why | Cost |
|---|---|---|---|
| Daily loop | **Verilator ≥5.036 + cocotb** | Fastest edit-run cycle; Python testbenches let you reuse the numpy golden model directly | free |
| Lint | **Verible** + `verilator --lint-only -Wall` | You already have Verible configured | free |
| Second opinion | **DSim** (Altair, ex-Metrics) | Full SystemVerilog + UVM; catches what Verilator's 2-state, synthesisable-subset view hides | free individual licence |
| Pre-synthesis | **Vivado xsim** | Matches the semantics of the tool that will synthesise it | free (Standard) |
| Synthesis / P&R | Vivado Standard → PRO → Enterprise | | escalating |

### On DSim specifically

Altair acquired Metrics Design Automation; DSim is the successor to the old
Metrics cloud simulator. There is a **free individual licence** — single
concurrent simulation, generated from the DSim Cloud portal, valid 90 days and
renewable. That is enough for one person's regression. ([licensing docs](https://help.metrics.ca/support/solutions/articles/154000146984-licensing))

Two caveats:
- The DSim Studio VS Code extension is marked deprecated in the Marketplace, so
  drive it from the CLI and your existing VS Code setup rather than building a
  workflow around the extension.
- The 90-day renewal means it is not something to put in unattended CI.

Keep DSim as the *second* simulator, not the first. Its value is disagreement
with Verilator, and you only get that if both are running the same tests —
which is exactly what cocotb buys you.

### Verilator version note

cocotb 2.x requires **Verilator ≥ 5.036**. Distro packages are usually far
older (Ubuntu 24.04 ships 5.020). Build from source:

```bash
git clone --depth 1 --branch v5.038 https://github.com/verilator/verilator
cd verilator && autoconf && ./configure --prefix=/usr/local && make -j$(nproc) && sudo make install
```

On your EPYC box that is a couple of minutes.

---

## Verification strategy

The structure in `tb/` is deliberate and worth keeping as the design grows:

1. **A numpy golden model that is itself self-checking.** `partition_mac`
   asserts its bit-serial result equals `np.dot`. The model cannot drift from
   the maths without failing on its own.
2. **Bit-exact comparison, never tolerance.** Integer datapaths have integer
   answers. `assert got == exp`. The only tolerance in the suite is the
   int-vs-float softmax error, and that is *logged*, not asserted — it is a
   measurement, not a pass/fail.
3. **Corners first.** `test_corner_values` hits ±128/±127 because signed
   nibble-split MACs almost always break at exactly those points.
4. **Property tests over directed tests** where you can. Random vectors against
   the model find more than hand-written cases.

What to add as you go:
- A cycle-count assertion per operation. The paper's 33 % latency claim is
  checkable, and it is the kind of number a reviewer will ask you to back up.
- An accuracy harness: run the LUT softmax inside a real quantised model
  (the paper used TinyLlama + lm-evaluation-harness) and report task accuracy
  deltas. Reuse their exact benchmark set so the comparison is fair.
- Coverage once you have the sequencer — until then there is not much state to
  cover.

---

## Where there is actual research here

Re-implementation on its own is not a contribution. Three things in this paper
are soft enough to push on, and all three connect to your MoE tiering work:

1. **The constant-max simplification is fragile and they know it.** They
   replace the running max with `z_quant_max = 127` and then evaluate with
   *full-precision* tables, which hides the interaction. At realistic 8-bit
   table precision, rows whose scores sit far below 127 underflow — I measured
   125/256 non-zero entries at `score_scale = 0.05`. A per-tile or
   per-head coarse max (one extra register, no blocking pass) might recover
   most of the range for almost no cost. That is a clean, small, defensible
   result.

2. **The array shape is asserted, not justified.** 64×32 with 2 banks. No
   sensitivity study. For MoE decode — your actual thesis — the relevant
   question is what shape serves *expert-sized* GEMMs with the residency
   pattern you are already tracing. That is a direct bridge from this paper to
   `moe-tiering-thesis`.

3. **Sparsity is reported but not mechanised.** Fig. 8 gives TOPS/W at three
   activation-sparsity levels with no described gating. Either it is pure
   switching activity (in which case the claim is weaker than it reads), or
   there is a skip mechanism they did not document. Worth an email to Bas Ahn
   at TU/e — it is a fair question and the kind that often gets you the RTL.

On that last point: **email the authors.** Corporaal's group at TU/e releases
code reasonably often, the paper is recent, and "I am re-implementing your
architecture, here is what I reconstructed, did I get the sign handling right"
is a message researchers like receiving. Attach `01-spec-teardown.md`. Worst
case you get nothing; best case you get the RTL and skip M1 entirely.
