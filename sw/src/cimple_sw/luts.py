"""Exp / reciprocal look-up tables and the fixed-point helpers of the split softmax.

Torch (vectorised) versions of the table builders and integer helpers in the
golden model ``cimple_rtl.model.cimple_model`` (imported here as ``gm``).
Every integer helper is exact and is tested for equality (==) against its
``gm`` counterpart, so the PyTorch kernel in ``split_softmax.py`` can claim
bit-exactness with the RTL datapath for the "hw8" + "lut" configuration --
including the RTL's negative-rounding bias (see ``requantise``).

Paper: Ahn et al., "CIMple: Standard-cell SRAM-based CIM with LUT-based split
softmax for attention acceleration", arXiv:2604.15944v2, Sec. IV-B / Fig. 7.

  * [F] exp LUT is 256 x 8b, indexed by the 8b quantised score (Fig. 4).
  * [P] the per-row max is replaced by the constant z_quant_max (Sec. IV-B).
  * [P] a second LUT "M" approximates the reciprocal of the exp-sum (Sec. IV-B).
  * [A] table contents (the paper gives only sizes) follow docs/01-spec-teardown.md:
        EXP_LUT[i]   = round(255 * exp((int8(i) - z_quant_max) * score_scale))
        RECIP_LUT[m] = round(2**15 / (256 + m)),  m = 8 bits below the leading 1
  * [P] the accuracy study used "full-precision tables"; ``exp_precision="full"``
    gives float64 entries [A: float64, not PyTorch's default float32, so the
    constant-max offset exp((z - z_quant_max) * s) of a row whose max sits far
    below the head's absmax does not underflow: float32 goes subnormal below
    e^-87 and hits 0 below e^-104, float64 only below e^-745].
"""
from __future__ import annotations

import numpy as np
import torch

from cimple_rtl.model import cimple_model as gm

from .config import SoftmaxConfig

__all__ = [
    "exp_table", "recip_lut", "msb_pos", "mantissa8", "recip_apply",
    "requantise", "round_half_away", "score_index",
]

LUT_W = gm.LUT_W            # 8b LUT index / output width
RECIP_FRAC = gm.RECIP_FRAC  # 15 fractional bits in the reciprocal LUT
SUM_W = gm.SUM_W            # 24b exp-sum accumulator [A in gm]


# ---------------------------------------------------------------------------
# exp table
# ---------------------------------------------------------------------------
def _signed_codes(bits: int) -> torch.Tensor:
    """Signed value of every unsigned index pattern 0 .. 2**bits-1."""
    n = 1 << bits
    idx = torch.arange(n, dtype=torch.int64)
    return ((idx + n // 2) % n) - n // 2


def score_index(z: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed score -> unsigned LUT index (the raw bit pattern, like ``gm.u8``)."""
    return z.to(torch.int64) & ((1 << bits) - 1)


def _exp_lut_hw8_np(scale: float, bits: int, zqm: int) -> np.ndarray:
    """One 8b UQ0.8 exp table, computed with exactly the numpy ops of gm."""
    if bits == gm.DATA_W:
        return gm.build_exp_lut(scale, zqm)
    n = 1 << bits                                   # same formula, other width
    idx = np.arange(n, dtype=np.int64)
    z = ((idx + n // 2) % n) - n // 2
    e = np.rint(255.0 * np.exp((z - zqm) * scale))
    return np.clip(e, 0, 255).astype(np.int64)


def exp_table(score_scale, cfg: SoftmaxConfig) -> torch.Tensor:
    """Exp LUT(s): shape ``score_scale.shape + (2**cfg.score_bits,)``.

    Entry i is the LUT output for raw score pattern i (signed value s(i)):
      "full" -> float64 exp((s(i) - zq_max) * scale)  (see module doc [A])
      "hw8"  -> int64 round(255 * exp(...)) clipped to [0, 255]; for each scalar
                scale this is ``gm.build_exp_lut(scale)`` (same numpy ops, so
                bit-identical; one table per *distinct* scale).
    """
    bits, zqm = cfg.score_bits, cfg.z_quant_max
    s = torch.as_tensor(score_scale, dtype=torch.float64)
    n = 1 << bits
    if cfg.exp_precision == "full":
        z = _signed_codes(bits).to(torch.float64)
        return torch.exp((z - zqm) * s.unsqueeze(-1))         # float64, no fp32 underflow
    if cfg.exp_precision != "hw8":
        raise ValueError(f"unknown exp_precision {cfg.exp_precision!r}")
    uniq, inv = torch.unique(s.reshape(-1), return_inverse=True)
    tabs = np.stack([_exp_lut_hw8_np(float(u), bits, zqm) for u in uniq.tolist()]) \
        if uniq.numel() else np.zeros((0, n), dtype=np.int64)
    return torch.from_numpy(tabs)[inv].reshape(*s.shape, n)


# ---------------------------------------------------------------------------
# reciprocal LUT and the leading-one normalisation
# ---------------------------------------------------------------------------
def recip_lut() -> torch.Tensor:
    """RECIP_LUT (256,) int64 == ``gm.build_recip_lut()``."""
    return torch.from_numpy(gm.build_recip_lut().astype(np.int64))


def msb_pos(x: torch.Tensor) -> torch.Tensor:
    """Position of the leading 1 (0 for x == 0); exact for 0 <= x < 2**63."""
    x = torch.as_tensor(x, dtype=torch.int64)
    if (x < 0).any():
        raise ValueError("msb_pos expects non-negative integers")
    p = torch.zeros_like(x)
    for k in (32, 16, 8, 4, 2, 1):                  # binary search on the bit index
        t = x >> k
        hit = t > 0
        x = torch.where(hit, t, x)
        p = p + hit.to(torch.int64) * k
    return p


def mantissa8(x: torch.Tensor) -> torch.Tensor:
    """The 8 bits below the leading 1 (zero-padded when x < 256), as ``gm.mantissa8``."""
    x = torch.as_tensor(x, dtype=torch.int64)
    p = msb_pos(x)
    down = x >> (p - LUT_W).clamp_min(0)
    up = x << (LUT_W - p).clamp_min(0)
    return torch.where(p >= LUT_W, down, up) & 0xFF


def recip_apply(sum_int: torch.Tensor):
    """Integer exp-sum -> (M, shift) with partial * M >> shift ~= partial / sum.

    Exactly gm.softmax_row: M = RECIP_LUT[mantissa8(s)],
    shift = RECIP_FRAC - LUT_W + msb_pos(s). For s == 0 this gives M = 128,
    shift = 7 (and the partial is 0, so the output is 0), as in gm.
    """
    s = torch.as_tensor(sum_int, dtype=torch.int64)
    M = recip_lut()[mantissa8(s)]
    shift = RECIP_FRAC - LUT_W + msb_pos(s)
    return M, shift


# ---------------------------------------------------------------------------
# requantisation / rounding
# ---------------------------------------------------------------------------
def requantise(acc_int, scale_int, shift) -> torch.Tensor:
    """sat8(rnd(acc * scale / 2**shift)) == ``gm.quantise``, vectorised.

    rnd is what gm.quantise / rtl/quant_unit.sv actually compute (their
    comments say round-half-away, but the arithmetic shift is a floor):
      x >= 0 : floor(x + 1/2)  -- round-half-away, correct;
      x <  0 : floor(x - 1/2)  -- correct only on exact ties (x = k + 1/2);
               every other negative x, exact integers included, comes out
               1 LSB too negative (e.g. -4/2 -> -3, -5/4 -> -2).
    [A] kept bit-exact with gm (the ground truth), so the "hw8" + "lut" kernel
    carries a -1 LSB bias on negative outputs; the "hw8" + "exact" recip path
    is a true round-half-away and therefore disagrees on negatives.

    All arguments broadcast; int64 arithmetic. Requires 0 <= shift <= 62 and
    |acc * scale| < 2**62 (gm uses unbounded Python ints; the softmax path stays
    below 2**39: |partial| < 2**31 for Lk < 2**16, M <= 128).
    """
    acc = torch.as_tensor(acc_int, dtype=torch.int64)
    scl = torch.as_tensor(scale_int, dtype=torch.int64)
    sh = torch.as_tensor(shift, dtype=torch.int64)
    if (sh < 0).any() or (sh > 62).any():
        raise ValueError("shift must be in [0, 62]")
    prod = acc * scl
    half = torch.where(sh > 0, torch.ones_like(sh) << (sh - 1).clamp_min(0),
                       torch.zeros_like(sh))
    r = torch.where(prod >= 0, prod + half, prod - half)
    return (r >> sh).clamp(-128, 127)               # >> is arithmetic (floor), like Python


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Float round-half-away-from-zero (the quantiser's *intended* rounding), same dtype.

    floor + exact fractional compare, so 0.49999999999999994 rounds to 0
    (``floor(|x| + 0.5)`` would give 1).
    """
    a = x.abs()
    f = torch.floor(a)
    return torch.sign(x) * (f + (a - f >= 0.5).to(x.dtype))
