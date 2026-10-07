# Resources

Ordered by how useful I think they actually are to you, not by category.

## 1. The one codebase to read before writing more RTL

**ITA — Integer Transformer Accelerator**, PULP Platform / ETH Zürich
<https://github.com/pulp-platform/ITA>

This is the closest open-source thing to what you are building and it is
*reference [18]* in the CIMple paper itself. INT8 attention + softmax
accelerator, SystemVerilog RTL, dual-licensed Apache-2.0 (software) /
Solderpad-0.51 (hardware). Crucially it ships `PyITA/` — a Python golden model
and test-vector generator — which is exactly the architecture this repo's
`model/` + `tb/` follows.

Read specifically:
- their integer-only streaming softmax, and compare it to CIMple's split-LUT
  approach — these are two different answers to the same problem and the
  comparison is a paper section on its own;
- how they structure test-vector generation from an ONNX graph;
- their Bender-based build. (It targets ModelSim; you will want to add a
  Verilator/cocotb path, which is a contained, useful exercise.)

Paper: Islamoglu et al., *"ITA: An Energy-Efficient Attention and Softmax
Accelerator for Quantized Transformers"*, ISLPED 2023 —
[arXiv:2307.03493](https://arxiv.org/html/2307.03493v2) ·
[slides](https://pulp-platform.org/docs/islped2023/ITA_GamzeIslamoglu-1.pdf)

Follow-up with the full heterogeneous system around ITA:
[arXiv:2408.02473](https://arxiv.org/pdf/2408.02473).

## 2. The bitcell CIMple is built on

Mori et al., *"A 4nm 6163-TOPS/W/b SRAM-based digital-computing-in-memory macro
supporting bit-width flexibility and simultaneous MAC and weight update"*,
ISSCC 2023 — reference [7]. This is where the 8T cell + OAI multiplier comes
from. If you want to understand *why* the OAI doubles as the bank selector,
this is the paper, not CIMple.

The same TSMC group's other macros are the lineage CIMple positions against:
Dong ISSCC'20 [4], Fujiwara ISSCC'22 [6] and ISSCC'24 [8], Lee VLSI'22 [9].
Reading two of these is enough to see the design space.

## 3. Direct competitors to benchmark against

- **TranCIM** — Tu et al., JSSC 58(6) 2023 [10]. Full-digital bitline-transpose
  CIM sparse transformer accelerator. The closest prior art.
- **MulTCIM** — Tu et al., JSSC 59(1) 2024 [21]. Attention-token-bit hybrid
  sparsity.
- **CIMFormer** — Guo et al., JSSC 2024 [22]. Systolic CIM array transformer.

These three plus ITA are your comparison table.

## 4. Softmax hardware (the LUT side)

- Dong, Zhu, Ma, *"Hardware implementation of softmax function based on
  piecewise LUT"*, IWOFC 2019 [19] — the baseline LUT approach CIMple improves on.
- Vasyltsov & Chang, *"Efficient softmax approximation for DNNs with attention"*,
  [arXiv:2111.10770](https://arxiv.org/abs/2111.10770) [20].
- Milakov & Gimelshein, *"Online normalizer calculation for softmax"*,
  [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) [15] — the running-max
  trick CIMple deliberately *abandons*. Read it to understand what they gave up.
- I-BERT, Kim et al., ICML 2021 [23] — integer-only nonlinearities.

## 5. Tooling

- **Verilator** — <https://verilator.org>. Build ≥5.036 from source; distro
  packages lag. ([Wikipedia overview](https://en.wikipedia.org/wiki/Verilator))
- **cocotb** — <https://docs.cocotb.org>. Note the 1.x → 2.x API break
  (`units=` → `unit=`, `BinaryValue` → `LogicArray`, `cocotb.fork` removed).
  This repo targets 2.x.
- **DSim** (Altair, ex-Metrics) — free individual licence, single concurrent
  sim, 90-day renewable:
  [licensing docs](https://help.metrics.ca/support/solutions/articles/154000146984-licensing) ·
  [overview](https://rydevinc.com/2024/10/15/dsim-a-saas-solution-for-your-rtl-verification-needs/) ·
  [AUR package](https://aur.archlinux.org/packages/dsim).
  The [DSim Studio VS Code extension](https://marketplace.visualstudio.com/items?itemName=AltairEngineering.dsim-studio)
  is marked deprecated — use the CLI.
- **Bender** — <https://github.com/pulp-platform/bender>. Dependency manager for
  HDL. Worth adopting early if you expect to pull in ITA or PULP IP; much less
  painful than retrofitting it later.

## 6. If you later want the ASIC branch

- **OpenRAM** — <https://github.com/VLSIDA/OpenRAM>. Open SRAM compiler, sky130
  and gf180 support. ([PyPI](https://pypi.org/project/openram/))
- **Cornell ECE 5745 SRAM tutorials** — the clearest free explanation of how an
  SRAM generator plugs into a synthesis flow:
  [section](https://cornell-ece5745.github.io/ece5745-S05-srams/) ·
  [tutorial](https://cornell-ece5745.github.io/ece5745-tut8-sram/)
- **OpenROAD / OpenLane** for the place-and-route half.

Remember what `02-roadmap.md` says about this branch: sky130 is 130 nm, so the
efficiency numbers will not be comparable to the paper's 28 nm. It is a
learning exercise and a flow-validation exercise, not a reproduction.

## 7. People to contact

Bas Ahn and Henk Corporaal, Electronic Systems, TU Eindhoven. Addresses are on
the paper's first page. The work is funded under EU Horizon Europe Convolve
(grant 101070374), which usually carries some openness expectation — worth
mentioning if you ask about code.
