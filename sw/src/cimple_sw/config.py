"""Configuration for the CIMple softmax / attention numerics.

One frozen dataclass so every layer of the stack (the torch kernel, the HF
attention hook, the eval CLI, the mapping simulator) agrees on what a "mode"
means. Paper references: Ahn et al., arXiv:2604.15944v2, Sec. IV-B and V-B.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

SoftmaxMode = Literal["float", "split_lut"]
ExpPrecision = Literal["full", "hw8"]
RecipMode = Literal["exact", "lut"]
ScoreScaleMode = Literal["dynamic", "static"]
ZMaxMode = Literal["const", "row"]


@dataclass(frozen=True)
class SoftmaxConfig:
    """How attention probabilities are computed.

    mode
        "float"     -- stock floating-point softmax (the paper's baseline:
                       "PyTorch's LogSoftmax" on the INT8 model).
        "split_lut" -- CIMple's LUT-based split softmax (Fig. 7): INT8 scores,
                       exp LUT for the numerator, exp-sum + reciprocal for the
                       denominator, applied after the e @ V accumulation.
    score_bits
        Bit width of the quantised score fed to the exp LUT (paper: 8, so the
        LUT has 2**8 = 256 entries).
    score_scale
        "dynamic" -- per (batch, head) scale = max |unmasked score| / 127,
                     computed on the fly (what a PyTorch evaluation does).
                     [A] symmetric absmax: when a head's negative tail is
                     larger than its positive max, every z sits far below
                     z_quant_max and, with zmax="const" + hw8 tables, the
                     whole head can underflow to exp-sum 0.
        "static"  -- a fixed per-layer scale from calibration (what the
                     hardware quantisation unit's scale register holds).
    zmax
        "const" -- subtract z_quant_max = 2**(score_bits-1) - 1 (the paper's
                   simplification: no per-row max pass).
        "row"   -- subtract the per-row max of the quantised score (classic
                   safe softmax; an ablation, not the paper's design).
    exp_precision
        "full" -- full-precision exp table (float64 entries [A], so the
                  constant-max offset does not underflow). The paper evaluates
                  accuracy this way: "we chose full-precision tables to
                  isolate the effect of the softmax approximation".
        "hw8"  -- 8b UQ0.8 table exactly as the RTL / golden model builds it:
                  round(255 * exp((z - zmax) * s)), clipped to [0, 255].
    recip
        "exact" -- 1 / sum in floating point. [A] The paper reads M from a
                   full-precision reciprocal LUT but gives neither its index
                   width nor its entry format, so that table is modelled as
                   exact 1/sum.
        "lut"   -- 256-entry reciprocal LUT indexed by the 8 bits below the
                   leading one of the integer exp-sum, as in the golden model
                   (only meaningful with exp_precision="hw8"; with "full" it
                   normalises the float sum's mantissa the same way).
    quantize_qkv
        Quantise Q, K, V to INT8 (per (batch, head) symmetric) before the two
        matmuls, as the CIM core only computes INT8 x INT8. Honoured in both
        modes (in "float" mode the fp32 softmax runs on the INT8 Q/K/V).
    quantize_out
        Requantise the attention output to INT8 in V's scale (the quantiser
        after the final M multiply in Fig. 7). Needs quantize_qkv (no V scale
        otherwise). In "float" mode only the HF hook applies it
        (``hf_attention``); ``split_softmax.cimple_attention`` ignores it there.
    """

    mode: SoftmaxMode = "split_lut"
    score_bits: int = 8
    score_scale: ScoreScaleMode = "dynamic"
    zmax: ZMaxMode = "const"
    exp_precision: ExpPrecision = "full"
    recip: RecipMode = "exact"
    quantize_qkv: bool = True
    quantize_out: bool = True

    @property
    def z_quant_max(self) -> int:
        return (1 << (self.score_bits - 1)) - 1

    def with_(self, **kw) -> "SoftmaxConfig":
        return replace(self, **kw)


# Named presets used by the CLI and the report.
PRESETS: dict[str, SoftmaxConfig] = {
    # INT8 model, stock softmax -- the paper's baseline
    "float": SoftmaxConfig(mode="float", quantize_qkv=False, quantize_out=False),
    # matched baseline [A]: same INT8 Q/K/V + INT8 output as "cimple", fp32
    # softmax -- "cimple" minus this isolates the softmax approximation alone
    # (the paper's stated intent, Sec. V-B); "cimple" minus "float" also
    # includes the attention INT8 quantisation, which the paper's baseline may
    # or may not have had (unspecified)
    "float_int8attn": SoftmaxConfig(mode="float", quantize_qkv=True, quantize_out=True),
    # the paper's evaluated configuration: INT8 scores, full-precision tables
    "cimple": SoftmaxConfig(),
    # same, but the 8b exp table + reciprocal LUT of the RTL (bit-exact)
    "cimple_hw8": SoftmaxConfig(exp_precision="hw8", recip="lut"),
    # ablation: per-row max instead of the constant z_quant_max, hw8 tables
    "cimple_hw8_rowmax": SoftmaxConfig(exp_precision="hw8", recip="lut", zmax="row"),
}
