"""
Bit-accurate golden model for the CIMple datapath re-implementation.

Everything here is integer arithmetic that mirrors the RTL exactly, so the
cocotb tests can compare cycle results against it with ==, not isclose().

Reference: Ahn et al., "CIMple: Standard-cell SRAM-based CIM with LUT-based
split softmax for attention acceleration", arXiv:2604.15944v2.
"""
from __future__ import annotations

import numpy as np

# ---- geometry (must match rtl/cimple_pkg.sv) -------------------------------
N_ROWS = 64
N_PART = 32
N_BANKS = 2
DATA_W = 8
NIB_W = 4
TREE_W = 10
COMB_W = 15
MAC_W = 23
N_LANES = 4
MUX_PHASES = N_PART // N_LANES
ACC_W = 32
QOUT_W = 8
LUT_DEPTH = 256
LUT_W = 8
SUM_W = 24
RECIP_FRAC = 15


def s8(x):
    """Reinterpret the low 8 bits as a signed INT8."""
    return ((np.asarray(x).astype(np.int64) + 128) % 256) - 128


def u8(x):
    """Reinterpret a signed INT8 as its unsigned 8-bit pattern (LUT index)."""
    return np.asarray(x).astype(np.int64) % 256


# ---------------------------------------------------------------------------
# CIM array
# ---------------------------------------------------------------------------
def nibble_split(w):
    """INT8 weight -> (signed MSB nibble, unsigned LSB nibble).

    w = msb*16 + lsb exactly, for the whole INT8 range.
    """
    raw = u8(w)
    lsb = raw & 0xF
    msb = ((raw >> 4) + 8) % 16 - 8          # signed 4-bit
    assert np.all(msb * 16 + lsb == s8(w))
    return msb, lsb


def partition_mac(weights, acts):
    """One partition, one full 8-cycle bit-serial pass.

    weights : (N_ROWS,) INT8      -- resident in the active bank
    acts    : (N_ROWS,) INT8      -- broadcast activation vector
    returns : the 23b accumulator value, and the per-cycle trace.
    """
    weights = s8(weights)
    acts = s8(acts)
    msb, lsb = nibble_split(weights)
    a_raw = u8(acts)

    acc = 0
    trace = []
    for k in range(DATA_W - 1, -1, -1):          # MSB first: bit 7 .. bit 0
        xb = (a_raw >> k) & 1
        sum_msb = int(np.sum(msb * xb))
        sum_lsb = int(np.sum(lsb * xb))
        assert -(1 << (TREE_W - 1)) <= sum_msb < (1 << (TREE_W - 1))
        assert 0 <= sum_lsb < (1 << TREE_W)
        comb = (sum_msb << NIB_W) + sum_lsb
        assert -(1 << (COMB_W - 1)) <= comb < (1 << (COMB_W - 1))
        acc = -comb if k == DATA_W - 1 else (acc << 1) + comb
        trace.append((k, sum_msb, sum_lsb, comb, acc))

    assert acc == int(np.dot(weights.astype(np.int64), acts.astype(np.int64)))
    assert -(1 << (MAC_W - 1)) <= acc < (1 << (MAC_W - 1))
    return acc, trace


def core_mac(weights, acts):
    """weights: (N_PART, N_ROWS) INT8 -> (N_PART,) accumulators."""
    return np.array([partition_mac(weights[p], acts)[0] for p in range(N_PART)],
                    dtype=np.int64)


def out_mux(acc, phase):
    """The 32->4 output select: lane l, phase k -> partition l*8 + k."""
    return [int(acc[l * MUX_PHASES + phase]) for l in range(N_LANES)]


# ---------------------------------------------------------------------------
# Quantiser
# ---------------------------------------------------------------------------
def quantise(acc, scale, shift):
    """sat8(round_half_away(acc * scale / 2**shift))."""
    prod = int(acc) * int(scale)
    if shift == 0:
        r = prod
    elif prod >= 0:
        r = prod + (1 << (shift - 1))
    else:
        r = prod - (1 << (shift - 1))
    v = r >> shift                      # arithmetic shift (Python ints)
    return max(-128, min(127, v))


# ---------------------------------------------------------------------------
# LUT-based split softmax
# ---------------------------------------------------------------------------
def build_exp_lut(score_scale: float, z_quant_max: int = 127):
    """EXP_LUT[i] = round(255 * exp((s8(i) - z_quant_max) * score_scale)).

    The table absorbs three things so the hardware does none of them:
      * the dequantisation of the INT8 score,
      * the subtraction of the constant max (the paper's key simplification:
        z_quant_max replaces the per-row running max),
      * the requantisation of the result to UQ0.8.
    """
    idx = np.arange(LUT_DEPTH)
    z = s8(idx)
    zdot = (z - z_quant_max) * score_scale
    e = np.rint(255.0 * np.exp(zdot))
    return np.clip(e, 0, 255).astype(np.int64)


def build_recip_lut():
    """RECIP_LUT[m] = round(2**RECIP_FRAC / (256 + m)), clipped to 8 bits.

    Index m is the 8 bits below the leading 1 of the exp-sum, so the value
    being inverted is 1.m in UQ1.8 -> (256+m)/256.
    """
    m = np.arange(LUT_DEPTH)
    v = np.rint((1 << RECIP_FRAC) / (256 + m))
    return np.clip(v, 0, 255).astype(np.int64)


def msb_pos(x: int) -> int:
    return 0 if x == 0 else x.bit_length() - 1


def mantissa8(x: int) -> int:
    p = msb_pos(x)
    if p >= LUT_W:
        return (x >> (p - LUT_W)) & 0xFF
    return (x << (LUT_W - p)) & 0xFF


def softmax_row(scores, values, exp_lut, recip_lut):
    """One full split-softmax row.

    scores : (N,) INT8  quantised QK^T
    values : (N,) INT8  the V vector this row is reduced against
    returns: (e, sum, M, shift, partial, final INT8 score)
    """
    e = np.array([int(exp_lut[int(u8(z))]) for z in scores], dtype=np.int64)
    s = int(e.sum())
    assert s < (1 << SUM_W), "exp-sum overflow -- widen SUM_W or shorten the row"

    partial = int(np.dot(e, s8(values).astype(np.int64)))
    m = mantissa8(s)
    M = int(recip_lut[m])
    shift = RECIP_FRAC - LUT_W + msb_pos(s)
    final = quantise(partial, M, shift)
    return dict(e=e, sum=s, M=M, shift=shift, partial=partial, score=final)


def reference_softmax_row(scores, values, score_scale, z_quant_max=127):
    """Float reference, for measuring the approximation error of the above."""
    z = (s8(scores).astype(np.float64) - z_quant_max) * score_scale
    p = np.exp(z)
    p /= p.sum()
    return float(np.dot(p, s8(values).astype(np.float64)))
