# cimple-rtl — a re-implementation of the CIMple attention datapath

Open-source-toolchain re-implementation of the compute-in-memory datapath from
Ahn et al., *"CIMple: Standard-cell SRAM-based CIM with LUT-based split softmax
for attention acceleration"* (arXiv:2604.15944v2, TU Eindhoven).

**Scope of this repo:** CIM core + requantiser + LUT split softmax, verified
bit-exact against a numpy golden model. The mapping/sequencing FSM
(encoder-only / decoder-only attention) is the next milestone, not here yet.

**What this is not:** a reproduction of the paper's 26.1 TOPS/W or
2.31 TOPS/mm². Those come from a custom 8T standard-cell bitcell in ST 28 nm
FD-SOI and cannot be reproduced without that PDK. See `docs/02-roadmap.md`.

## Layout

Everything below lives under `src/cimple_rtl/` (`model/` is an importable
subpackage: `from cimple_rtl.model import cimple_model`).

```
rtl/
  cimple_pkg.sv       parameters; every width traced to the paper's Fig. 4
  cim_macro.sv        16-bitcell / 8-OAI macro -- THE ONLY non-faithful module
  cim_partition.sv    64 macros + adder trees + <<4 + <<1&ACC shift-accumulator
  cim_core.sv         32 partitions + 32->4 output mux under a 3b counter
  quant_unit.sv       32b -> 8b requantiser, round-half-away, saturating
  lut_rom.sv          $readmemh ROM
  split_softmax.sv    exp LUT + exp-sum + reciprocal LUT + final scale
model/
  cimple_model.py     self-checking numpy golden model
  gen_luts.py         emits exp_lut.hex / recip_lut.hex
tb/
  test_*.py           cocotb tests, bit-exact against the model
  Makefile
docs/
  01-spec-teardown.md every width and its evidence; assumptions flagged
  02-roadmap.md       milestones, tool path, and where the research is
  03-resources.md     papers, code, tools
```

## Quick start

```bash
# Verilator >= 5.036 required by cocotb 2.x -- build from source if your
# distro package is older (Ubuntu 24.04 ships 5.020).
uv sync

make lint          # verilator --lint-only, both top levels
make test          # generate the LUTs, then run the whole suite

# or piecemeal
uv run python -m cimple_rtl.model.gen_luts --score-scale 0.05   # -> build/*.hex
uv run make -C src/cimple_rtl/tb MODULE=test_cim_partition TOPLEVEL=cim_partition   # just one
uv run make -C src/cimple_rtl/tb WAVES=1 MODULE=test_cim_core TOPLEVEL=cim_core     # with VCD
```

Current status: **9 tests, all passing** on Verilator 5.038 + cocotb 2.1.
The softmax test also logs the INT8-LUT-vs-float error (mean 1.43 LSB,
max 6.03 LSB at `score_scale = 0.05`) as a measurement rather than an assert.

## Design decisions you should review

Three things the paper does not specify, decided here. Read
`docs/01-spec-teardown.md` §2 and §5 before you build on them:

1. **Sign handling** — MSB nibble signed, LSB nibble unsigned, activations
   MSB-first with the first `<<1 & ACC` cycle subtracting.
2. **LUT contents** — the exp table absorbs dequantisation, the `−z_quant_max`
   offset, and requantisation; the reciprocal table inverts a normalised
   mantissa.
3. **Bank policy** — both banks writable, one readable; no double-buffering
   logic yet because the paper does not say what it is for.

## Parameterisation

`rtl/cimple_pkg.sv` is the single source of truth and the golden model mirrors
it. For first FPGA bring-up, shrink `N_PART` to 8 and `N_ROWS` to 16 — the
whole tree and the full test suite scale with it.
