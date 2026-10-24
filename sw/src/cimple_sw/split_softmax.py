"""CIMple's LUT-based split softmax as a PyTorch attention kernel.

This is the kernel the paper "wrote in PyTorch" for its accuracy study
(Ahn et al., arXiv:2604.15944v2, Sec. V-B): "a LUT-based split softmax
approach quantized to int8 ... evaluat[ed] with a pre-trained transformer
model quantized to int8, TinyLlama". The flow is Fig. 7 / Sec. IV-B:

    z      = quant8(Q K^T * scaling)                 INT8 score
    e_j    = EXP_LUT[z_j]   (= exp(z_j - z_quant_max) in score units)
    sum    = sum_j e_j                                denominator (exp-sum)
    partial= sum_j e_j * V_j                          numerator, on the CIM core
    out    = quant8(partial * M(sum))                 M ~= 1/sum (recip LUT)

The two simplifications of the paper are both here:
  * [P] constant max: z_quant_max = 2**(bits-1)-1 replaces the per-row max
    (``cfg.zmax="row"`` is the safe-softmax ablation, not the paper's design);
  * [P] split numerator / denominator: e streams into e @ V while the sum
    accumulates; the reciprocal is applied once per output.

Exactness. With ``exp_precision="hw8"``, ``recip="lut"``, ``quantize_out`` and
INT8 V, ``split_softmax_int`` is bit-exact with ``gm.softmax_row`` (tested with
==), including gm's output rounding: floor(x + 1/2) for x >= 0 but
floor(x - 1/2) for x < 0, i.e. a -1 LSB bias on every non-tie negative output
(``luts.requantise``; e.g. a row averaging V = -1 gives -2). Integer matmuls run in float64 BLAS: [D] |e*v| <= 255*128 < 2**15, so
every partial sum is an integer < 2**53 for Lk < 2**38 and therefore exact in
any summation order. INT8 Q K^T runs in float32: [D] |q*k| <= 2**14, sums
<= 2**24 for D <= 1024, again exact.

Assumptions (the paper does not specify them):
  * [A] Q, K, V quantised symmetric INT8 per (batch, head), scale = absmax/127,
    half-to-even rounding (``gm.to_int8``) -- per-tensor granularity is what a
    single scale register per CIM tile implies; per-head keeps heads independent.
  * [A] Score quantiser = float ``rint(score / s)`` + saturation (a PyTorch
    evaluation), not the integer (scale, shift) requantiser of the RTL. They
    differ by up to 2 LSB on ~30% of ordinary (non-tie) inputs: the RTL
    rounds negatives an extra LSB toward -inf (see ``luts.requantise``) and
    approximates the ratio 1/s with a 16b scale (``gm.ratio_to_scale_shift``),
    which flips values near a rounding boundary.
  * [A] Dynamic score scale s = max |kept score| / z_quant_max per (b, h)
    (symmetric absmax, like Q/K/V). If a head's negative tail is larger than
    its positive max, every z of the head sits far below z_quant_max; with
    hw8 tables + const max the *whole head* then underflows to exp-sum 0 (one
    key with score -37 vs a max of +5 is enough). A positive-side scale
    (max kept score / z_quant_max, e.g. via ``score_scale="static"``) avoids
    that; it is not the default so as to match the calibrated absmax scales.
  * [A] Masked keys contribute e = 0 (the hardware simply never streams them).
    Rows whose kept exps are all 0 (fully masked, or underflow of the constant
    max offset) output 0 -- exactly what gm.softmax_row returns for sum == 0.
  * [A] "full" tables hold float64 entries (not PyTorch's default float32:
    an fp32 table zeroes rows whose max is > ~104 logits below the head's
    absmax); sums and e @ V are float64.

Known limitations:
  * Per-(b, h) dynamic scales (Q/K/V absmax, dynamic score scale) are taken
    over the whole sequence, so in a causal prefill row t's *scales* depend on
    tokens > t (its attention weights never do: masked e are exactly 0). Use
    ``score_scale="static"`` and ``quantize_qkv=False`` for strict causality.
  * hw8 + const max: a row whose max score sits far below z_quant_max
    underflows to exp-sum 0 and outputs 0 -- the design's real weakness
    (measured on random scores: 3% dead rows at score std 2, 80% at std 5).
    The absmax score scale makes this worse for heads with a large negative
    tail (see the [A] above). "full" tables only underflow when a row's max
    is > ~745 logits below the head's absmax (float64 range).
"""
from __future__ import annotations

import torch

from cimple_rtl.model import cimple_model as gm

from .config import SoftmaxConfig
from .luts import exp_table, recip_apply, recip_lut, requantise, round_half_away, score_index

__all__ = ["split_softmax_int", "cimple_attention", "observe_score_absmax", "float_attention"]

QMAX8 = 127


# ---------------------------------------------------------------------------
# the integer split softmax (Fig. 7)
# ---------------------------------------------------------------------------
def _expand_keep(keep, shape):
    return None if keep is None else torch.as_tensor(keep, dtype=torch.bool).expand(shape)


def _float_recip(s: torch.Tensor, cfg: SoftmaxConfig) -> torch.Tensor:
    """1/s (float64), or its recip-LUT approximation; 0 where s == 0.

    "lut" on a float sum [A]: normalise s = 1.f * 2**p (frexp), take the 8
    truncated bits of f as the LUT index m, and 1/s ~= RECIP_LUT[m] * 2**-(7+p)
    -- the float analogue of gm's ``partial * M >> (RECIP_FRAC - 8 + p)``.
    """
    pos = s > 0
    if cfg.recip == "exact":
        r = 1.0 / torch.where(pos, s, torch.ones_like(s))
    elif cfg.recip == "lut":
        mant, ex = torch.frexp(torch.where(pos, s, torch.ones_like(s)))   # s = mant*2**ex
        p = ex.to(torch.int64) - 1                                        # leading-one pos
        m = torch.floor((mant * 2.0 - 1.0) * 256.0).to(torch.int64)       # exact: power-of-2 scale
        M = recip_lut()[m].to(torch.float64)
        r = torch.ldexp(M, -(gm.RECIP_FRAC - gm.LUT_W + p))
    else:
        raise ValueError(f"unknown recip {cfg.recip!r}")
    return torch.where(pos, r, torch.zeros_like(r))


def split_softmax_int(z, v, keep, cfg: SoftmaxConfig, score_scale) -> dict:
    """Split softmax of INT8 scores against V, vectorised over all rows.

    z     : int tensor (..., Lq, Lk), signed scores in [-2**(b-1), 2**(b-1)-1]
    v     : (..., Lk, D) -- signed INT8 (integer dtype) or float (then ``out``
            is always the float ratio; there is no INT8 output scale for V)
    keep  : bool broadcastable to z (True = attend) or None
    score_scale : float or tensor broadcastable to z[..., :1, :1]
    returns dict(e, sum, partial, out):
      e       (..., Lq, Lk) LUT outputs (int64 for "hw8", float64 for "full"), 0 where masked
      sum     (..., Lq)     exp-sum (int64 / float64)
      partial (..., Lq, D)  e @ v (int64 when e and v are integer, else float64)
      out     (..., Lq, D)  INT8 int64 in V's units if cfg.quantize_out and v is
                            integer, else the float64 ratio partial * recip(sum).
    hw8 + lut + quantize_out: out = requantise(partial, M, shift) == gm.softmax_row.
    """
    bits, zqm = cfg.score_bits, cfg.z_quant_max
    n = 1 << bits
    z = torch.as_tensor(z).to(torch.int64)
    if z.numel() and (z.min() < -(n // 2) or z.max() > n // 2 - 1):
        raise ValueError(f"scores outside the signed {bits}b range")
    keep = _expand_keep(keep, z.shape)

    # constant max (paper) or per-row max of the kept scores (ablation)
    zi = z
    if cfg.zmax == "row":
        zk = z if keep is None else z.masked_fill(~keep, -(n // 2) - 1)
        rmax = zk.amax(-1, keepdim=True)
        rmax = torch.where(rmax < -(n // 2), torch.full_like(rmax, zqm), rmax)   # fully masked
        zi = z - rmax + zqm                     # row max -> zq_max; stays >= -(n//2)
    elif cfg.zmax != "const":
        raise ValueError(f"unknown zmax {cfg.zmax!r}")

    # numerator LUT: one table per distinct (batch, head) scale
    s = torch.as_tensor(score_scale, dtype=torch.float64)
    s = torch.broadcast_to(s, (*z.shape[:-2], 1, 1))
    tbl = exp_table(s, cfg).squeeze(-2)                       # (..., 1, n)
    tbl = tbl.expand(*z.shape[:-1], n)
    e = torch.gather(tbl, -1, score_index(zi, bits))
    if keep is not None:
        e = e.masked_fill(~keep, 0)

    v = torch.as_tensor(v)
    int_e = cfg.exp_precision == "hw8"
    int_v = not v.is_floating_point()
    if int_e:
        ssum = e.sum(-1)
        if ssum.numel() and ssum.max() >= (1 << gm.SUM_W):
            raise OverflowError("exp-sum overflow -- row longer than the 24b accumulator allows")
    else:
        ssum = e.to(torch.float64).sum(-1)
    partial = e.to(torch.float64) @ v.to(torch.float64)      # exact for integers (see module doc)
    if int_e and int_v:
        partial = partial.to(torch.int64)

    if cfg.quantize_out and int_v:
        if int_e and cfg.recip == "lut":
            M, shift = recip_apply(ssum)
            out = requantise(partial, M.unsqueeze(-1), shift.unsqueeze(-1))
        elif int_e:                              # exact integer round-half-away(p / s)
            sd = ssum.unsqueeze(-1)
            den = torch.where(sd > 0, sd, torch.ones_like(sd))
            q = (2 * partial.abs() + den) // (2 * den)
            out = torch.where(sd > 0, torch.sign(partial) * q, torch.zeros_like(q)).clamp(-128, 127)
        else:
            ratio = partial * _float_recip(ssum, cfg).unsqueeze(-1)
            out = round_half_away(ratio).clamp(-128, 127).to(torch.int64)
    else:
        if int_e and cfg.recip == "lut":         # partial * M * 2**-shift
            M, shift = recip_apply(ssum)
            r = torch.where(ssum > 0, torch.ldexp(M.to(torch.float64), -shift),
                            torch.zeros((), dtype=torch.float64))
        else:
            r = _float_recip(ssum.to(torch.float64), cfg)
        out = partial.to(torch.float64) * r.unsqueeze(-1)
    return dict(e=e, sum=ssum, partial=partial, out=out)


# ---------------------------------------------------------------------------
# attention wrappers
# ---------------------------------------------------------------------------
def _quant_per_head(x: torch.Tensor):
    """Symmetric INT8 per (..., head): codes (int64), scale (..., 1, 1) float64.

    rint half-to-even + clip, exactly ``gm.to_int8`` with scale = absmax / 127;
    an all-zero head gets scale 1 [A] (its codes are 0 either way).
    """
    xd = x.to(torch.float64)
    sc = xd.abs().amax(dim=(-2, -1), keepdim=True) / QMAX8
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    return torch.round(xd / sc).clamp(-128, 127).to(torch.int64), sc


def _masked_softmax(scores: torch.Tensor, keep) -> torch.Tensor:
    if keep is not None:
        scores = scores.masked_fill(~torch.as_tensor(keep, dtype=torch.bool), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(p, nan=0.0)          # fully masked row -> zeros


def float_attention(q, k, v, *, scaling, keep=None):
    """Reference softmax(q k^T * scaling) v in float32; masked rows give 0."""
    qf, kf, vf = q.float(), k.float(), v.float()
    p = _masked_softmax((qf @ kf.transpose(-2, -1)) * scaling, keep)
    return (p @ vf).to(q.dtype)


def observe_score_absmax(q, k, *, scaling, keep=None) -> torch.Tensor:
    """max |q k^T * scaling| over kept positions, per (B, H) -> (B, H) float32.

    Uses the float scores [A] (static calibration happens before the INT8
    Q/K quantisation is known; the difference is the Q/K rounding only).
    """
    sc = (q.float() @ k.float().transpose(-2, -1)) * scaling
    if keep is not None:
        sc = sc.masked_fill(~torch.as_tensor(keep, dtype=torch.bool), 0.0)
    return sc.abs().amax(dim=(-2, -1))


def cimple_attention(q, k, v, *, scaling, keep=None, cfg: SoftmaxConfig,
                     score_scale=None, return_aux=False):
    """Attention with CIMple's numerics. q (B,H,Lq,D), k/v (B,H,Lk,D) float.

    cfg.mode == "float"     : float_attention (on fake-quantised q/k/v if
                              cfg.quantize_qkv).
    cfg.mode == "split_lut" : INT8 q/k/v (if cfg.quantize_qkv; otherwise float
                              q/k/v and only the scores are quantised), INT8
                              scores, split_softmax_int, dequantise by V's scale.
    score_scale : required for cfg.score_scale == "static"; scalar or tensor
                  broadcastable to (B, H, 1, 1). Ignored for "dynamic".
    Returns out (B,H,Lq,D) in q.dtype, plus an aux dict if return_aux
    (z, e, sum, probs = e/sum as float32, score_scale (B,H,1,1)).
    """
    dtype = q.dtype
    B, H, Lq, D = q.shape
    keep_b = None if keep is None else torch.as_tensor(keep, dtype=torch.bool)

    if cfg.mode == "float":
        if cfg.quantize_qkv:
            q, k, v = (c.to(torch.float64) * sc for c, sc in map(_quant_per_head, (q, k, v)))
        out = float_attention(q.float(), k.float(), v.float(), scaling=scaling, keep=keep_b)
        if return_aux:
            return out.to(dtype), {}
        return out.to(dtype)
    if cfg.mode != "split_lut":
        raise ValueError(f"unknown mode {cfg.mode!r}")

    # (1) INT8 q, k, v
    if cfg.quantize_qkv:
        (qi, sq), (ki, sk), (vi, sv) = map(_quant_per_head, (q, k, v))
        mm = torch.float32 if D <= 1024 else torch.float64       # exact, see module doc
        dot = (qi.to(mm) @ ki.to(mm).transpose(-2, -1)).to(torch.float64)
        score = dot * (sq * sk * scaling)                        # (2) dequantised score
    else:
        vi, sv = v.float(), torch.ones((), dtype=torch.float64)
        score = ((q.float() @ k.float().transpose(-2, -1)) * scaling).to(torch.float64)

    # (3) score scale
    zqm = cfg.z_quant_max
    if cfg.score_scale == "dynamic":
        sabs = score if keep_b is None else score.masked_fill(~keep_b, 0.0)
        s = sabs.abs().amax(dim=(-2, -1), keepdim=True) / zqm
        s = torch.where(s > 0, s, torch.ones_like(s))
    elif cfg.score_scale == "static":
        if score_scale is None:
            raise ValueError('score_scale="static" needs a score_scale argument')
        s = torch.broadcast_to(torch.as_tensor(score_scale, dtype=torch.float64),
                               (B, H, 1, 1)).clone()
    else:
        raise ValueError(f"unknown score_scale mode {cfg.score_scale!r}")

    # (4) INT8 score, saturating
    lo, hi = -(1 << (cfg.score_bits - 1)), (1 << (cfg.score_bits - 1)) - 1
    z = torch.round(score / s).clamp(lo, hi).to(torch.int64)
    del score

    # (5) split softmax, (6) dequantise
    r = split_softmax_int(z, vi, keep_b, cfg, s)
    out = (r["out"].to(torch.float64) * sv).to(dtype)
    if not return_aux:
        return out
    ssum = r["sum"].to(torch.float64)
    den = torch.where(ssum > 0, ssum, torch.ones_like(ssum)).unsqueeze(-1)
    probs = (r["e"].to(torch.float64) / den).to(torch.float32)
    return out, dict(z=z, e=r["e"], sum=r["sum"], probs=probs, score_scale=s)
