"""Tests for the PyTorch LUT-based split softmax (paper Sec. IV-B, Fig. 7).

Bit-exact claims are checked with == against the golden model
``cimple_rtl.model.cimple_model``; approximation quality is measured against
float softmax attention and the measured numbers are printed (pytest -s).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cimple_rtl.model import cimple_model as gm
from cimple_sw import luts
from cimple_sw.config import PRESETS, SoftmaxConfig
from cimple_sw.split_softmax import (cimple_attention, float_attention,
                                     observe_score_absmax, split_softmax_int)

HW8 = PRESETS["cimple_hw8"]
RNG_SEED = 1234


def _rng():
    return np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# LUTs and integer helpers
# ---------------------------------------------------------------------------
SCALES = [0.05, 0.01, 0.2, 0.0, 1.0, 0.012345, 0.0731, 3.0e-4]


@pytest.mark.parametrize("scale", SCALES)
def test_exp_table_hw8_equals_gm(scale):
    t = luts.exp_table(scale, HW8)
    assert t.dtype == torch.int64 and t.shape == (256,)
    assert np.array_equal(t.numpy(), gm.build_exp_lut(scale))


def test_exp_table_hw8_tensor_scale_broadcast():
    s = torch.tensor([[0.05, 0.01, 0.2], [0.05, 0.0731, 0.01]], dtype=torch.float64)
    t = luts.exp_table(s, HW8)
    assert t.shape == (2, 3, 256)
    for i in range(2):
        for j in range(3):
            assert np.array_equal(t[i, j].numpy(), gm.build_exp_lut(float(s[i, j])))


def test_exp_table_full():
    cfg = PRESETS["cimple"]
    s = torch.tensor([0.05, 0.2], dtype=torch.float64).reshape(2, 1)
    t = luts.exp_table(s, cfg)
    assert t.dtype == torch.float64 and t.shape == (2, 1, 256)
    z = gm.s8(np.arange(256)).astype(np.float64)
    for i, sc in enumerate((0.05, 0.2)):
        ref = np.exp((z - 127) * sc)
        np.testing.assert_allclose(t[i, 0].numpy(), ref, rtol=1e-15, atol=0)
    assert t[0, 0, 127] == 1.0                      # index 127 = z_quant_max -> exp(0)


def test_full_table_no_fp32_underflow():
    """N3: a row whose max sits > ~104 logits below the head's absmax must not die."""
    t = luts.exp_table(1.0, PRESETS["cimple"])
    assert (t > 0).all() and t.min() == np.exp(-255.0)           # fp32 would give 0
    r = split_softmax_int(torch.tensor([[127, 0], [-10, -6]]), torch.tensor([[50], [-50]]),
                          None, PRESETS["cimple"], 1.0)
    assert (r["sum"] > 0).all()
    w = 1.0 / (1.0 + np.exp(4.0))                                 # softmax([-10, -6])[0]
    assert r["out"].tolist() == [[50], [round(50 * w - 50 * (1 - w))]]   # [[50], [-48]]


def test_cimple_large_logit_head_has_no_dead_rows():
    """N3 at attention level: one score ~115 used to kill 62/64 rows of the 'cimple' preset."""
    L = D = 64
    q, k, v = (torch.randn(1, 1, L, D, generator=torch.Generator().manual_seed(1)) for _ in range(3))
    q[..., 5, :] = k[..., 0, :] * 14
    keep = _causal(L)
    ref = float_attention(q, k, v, scaling=D ** -0.5, keep=keep)
    out, aux = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["cimple"],
                                return_aux=True)
    assert (aux["sum"] > 0).all()
    assert _cos(out, ref) > 0.99                                  # measured 0.9999 (fp32 table: 0.29)


def test_exp_table_other_bits():
    cfg = HW8.with_(score_bits=6)
    t = luts.exp_table(0.1, cfg)
    assert t.shape == (64,) and t[31] == 255 and t[32] == round(255 * np.exp(-63 * 0.1))


def test_recip_lut_equals_gm():
    assert np.array_equal(luts.recip_lut().numpy(), gm.build_recip_lut())


def _int_cases():
    rng = _rng()
    edge = [0, 1, 2, 3, 255, 256, 257, 511, 512, (1 << 24) - 1, 1 << 24, (1 << 31) - 1, (1 << 40) + 7]
    edge += [1 << p for p in range(0, 62)] + [(1 << p) - 1 for p in range(1, 62)]
    rand = rng.integers(0, 1 << 24, size=20000).tolist() + rng.integers(0, 1 << 62, size=5000).tolist()
    return edge + rand


def test_msb_pos_and_mantissa8_equal_gm():
    xs = _int_cases()
    t = torch.tensor(xs, dtype=torch.int64)
    assert luts.msb_pos(t).tolist() == [gm.msb_pos(x) for x in xs]
    assert luts.mantissa8(t).tolist() == [gm.mantissa8(x) for x in xs]
    with pytest.raises(ValueError):
        luts.msb_pos(torch.tensor([-1]))


def test_recip_apply_equals_gm():
    xs = [x for x in _int_cases() if x < (1 << gm.SUM_W)]
    M, sh = luts.recip_apply(torch.tensor(xs, dtype=torch.int64))
    lut = gm.build_recip_lut()
    assert M.tolist() == [int(lut[gm.mantissa8(x)]) for x in xs]
    assert sh.tolist() == [gm.RECIP_FRAC - gm.LUT_W + gm.msb_pos(x) for x in xs]


def test_requantise_equals_gm():
    rng = _rng()
    n = 30000
    acc = rng.integers(-(1 << 31), 1 << 31, size=n)
    acc[: n // 3] = rng.integers(-3000, 3000, size=n // 3)      # many in/near the int8 range
    scale = rng.integers(-300, 300, size=n)
    shift = rng.integers(0, 31, size=n)
    # exact ties: acc*scale = (2k+1) * 2**(shift-1), both signs
    k = rng.integers(-200, 200, size=2000)
    sh_t = rng.integers(1, 20, size=2000)
    acc = np.concatenate([acc, (2 * k + 1) << (sh_t - 1), [0, 0, -1, 1, 127, 128, -128, -129]])
    scale = np.concatenate([scale, np.ones(2000, dtype=np.int64), [5, 1, 1, 1, 1, 1, 1, 1]])
    shift = np.concatenate([shift, sh_t, [0, 0, 0, 0, 0, 0, 0, 0]])
    got = luts.requantise(torch.from_numpy(acc), torch.from_numpy(scale), torch.from_numpy(shift))
    ref = [gm.quantise(int(a), int(s), int(h)) for a, s, h in zip(acc, scale, shift)]
    assert got.tolist() == ref
    with pytest.raises(ValueError):
        luts.requantise(torch.tensor([1]), 1, -1)


def test_requantise_negative_rounding_is_floor_minus_half():
    """N1: pin what gm.quantise / quant_unit.sv really do, against exact Fractions.

    x >= 0 -> floor(x + 1/2) (round-half-away); x < 0 -> floor(x - 1/2), i.e. 1 LSB
    too negative on every negative non-tie. requantise stays bit-exact with gm.
    """
    import math
    from fractions import Fraction
    rng = _rng()
    acc = rng.integers(-5000, 5000, size=4000)
    scale = rng.integers(1, 300, size=4000)
    shift = rng.integers(1, 16, size=4000)
    got = luts.requantise(torch.from_numpy(acc), torch.from_numpy(scale), torch.from_numpy(shift)).tolist()
    n_bias = 0
    for a, sc, sh, g in zip(acc.tolist(), scale.tolist(), shift.tolist(), got):
        x = Fraction(a * sc, 1 << sh)
        rtl = math.floor(x + Fraction(1, 2)) if x >= 0 else math.floor(x - Fraction(1, 2))
        half_away = int(math.copysign(math.floor(abs(x) + Fraction(1, 2)), x))
        assert g == max(-128, min(127, rtl))
        is_tie = (x - Fraction(1, 2)).denominator == 1
        if x < 0 and not is_tie and -128 <= half_away - 1:
            assert g == half_away - 1                              # the documented -1 LSB bias
            n_bias += 1
        elif x >= 0 or is_tie:
            assert g == max(-128, min(127, half_away))
    assert n_bias > 400                                           # measured 620
    # the hand examples of the finding
    assert luts.requantise(torch.tensor([-4, -5, 5]), 1, torch.tensor([1, 2, 2])).tolist() == [-3, -2, 1]
    # ...and its effect on the bit-exact hw8 kernel: softmax of equal scores returns V,
    # but negative V comes out 1 LSB low (the hw8 'exact' recip path is true half-away)
    z = torch.full((1, 8), 127)
    for val, want in ((50, 50), (-50, -51), (1, 1), (-1, -2)):
        r = split_softmax_int(z, torch.full((8, 1), val), None, HW8, 0.05)
        assert r["out"].item() == want == gm.softmax_row(
            np.full(8, 127), np.full(8, val), gm.build_exp_lut(0.05), gm.build_recip_lut())["score"]
        assert split_softmax_int(z, torch.full((8, 1), val), None, HW8.with_(recip="exact"),
                                 0.05)["out"].item() == val


def test_round_half_away():
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, 0.49999999999999994, -2.4, 2.6, 0.0],
                     dtype=torch.float64)
    assert luts.round_half_away(x).tolist() == [1, 2, 3, -1, -2, 0, -2, 3, 0]


# ---------------------------------------------------------------------------
# split_softmax_int: bit-exact with gm.softmax_row
# ---------------------------------------------------------------------------
def _gm_rows(z, v, keep, scales, row_max=False):
    """Run gm.softmax_row on every (row, column), masked entries dropped."""
    R, Lq, L = z.shape
    D = v.shape[-1]
    recip = gm.build_recip_lut()
    out = np.zeros((R, Lq, D), dtype=np.int64)
    sums = np.zeros((R, Lq), dtype=np.int64)
    parts = np.zeros((R, Lq, D), dtype=np.int64)
    for r in range(R):
        lut = gm.build_exp_lut(float(scales[r]))
        for i in range(Lq):
            kk = keep[r, i]
            zr = z[r, i][kk]
            if row_max and zr.size:
                zr = zr - zr.max() + 127
            for c in range(D):
                res = gm.softmax_row(zr, v[r][kk, c], lut, recip)
                out[r, i, c], parts[r, i, c] = res["score"], res["partial"]
            sums[r, i] = res["sum"]
    return out, sums, parts


@pytest.mark.parametrize("L", [1, 2, 7, 33, 64, 128, 199, 300])
@pytest.mark.parametrize("row_max", [False, True])
def test_hw8_bit_exact_vs_gm(L, row_max):
    rng = np.random.default_rng(L * 7 + row_max)
    R, Lq, D = 6, 4, 3
    scales = np.array([0.05, 0.01, 0.2, 0.0731, 0.5, 0.02])
    z = rng.integers(-128, 128, size=(R, Lq, L))
    z[0, 0] = 127                                     # saturated row
    z[1, 1] = -128
    z[2, 2] = rng.integers(-128, -60, size=L)         # scale 0.2: underflows to sum 0
    v = rng.integers(-128, 128, size=(R, L, D))
    v[3] = 127                                        # all-max V: tests output saturation
    keep = rng.random((R, Lq, L)) < 0.8
    keep[4, 3] = False                                # fully masked row
    cfg = HW8.with_(zmax="row") if row_max else HW8
    res = split_softmax_int(torch.from_numpy(z), torch.from_numpy(v), torch.from_numpy(keep),
                            cfg, torch.from_numpy(scales).reshape(R, 1, 1))
    out, sums, parts = _gm_rows(z, v, keep, scales, row_max)
    assert res["out"].dtype == torch.int64
    assert np.array_equal(res["sum"].numpy(), sums)
    assert np.array_equal(res["partial"].numpy(), parts)
    assert np.array_equal(res["out"].numpy(), out)
    assert (res["out"][4, 3] == 0).all()
    if not row_max:
        assert (res["sum"][2, 2] == 0).all() and (res["out"][2, 2] == 0).all()


def test_hw8_no_mask_scalar_scale_vs_gm():
    rng = _rng()
    z = rng.integers(-128, 128, size=(2, 3, 50))
    v = rng.integers(-128, 128, size=(2, 50, 5))
    res = split_softmax_int(torch.from_numpy(z), torch.from_numpy(v), None, HW8, 0.03)
    out, _, _ = _gm_rows(z, v, np.ones(z.shape, bool), [0.03, 0.03])
    assert np.array_equal(res["out"].numpy(), out)


def _np_ref(z, v, keep, scale, recip_lut_mode):
    """float64 numpy reference for the non-bit-exact configurations."""
    e = np.exp((z.astype(np.float64) - 127) * scale)                 # float64 "full" table
    e = np.where(keep, e, 0.0)
    s = e.sum(-1)
    p = e @ v.astype(np.float64)
    if recip_lut_mode:
        mant, ex = np.frexp(np.where(s > 0, s, 1.0))
        m = np.floor((mant * 2 - 1) * 256).astype(np.int64)
        r = np.ldexp(gm.build_recip_lut()[m].astype(np.float64), -(7 + ex - 1))
    else:
        r = 1.0 / np.where(s > 0, s, 1.0)
    ratio = p * np.where(s > 0, r, 0.0)[..., None]
    return np.clip(np.sign(ratio) * np.floor(np.abs(ratio) + 0.5), -128, 127), ratio


@pytest.mark.parametrize("recip", ["exact", "lut"])
def test_full_precision_paths(recip):
    rng = _rng()
    z = rng.integers(-128, 128, size=(3, 5, 40))
    v = rng.integers(-128, 128, size=(3, 40, 4))
    keep = rng.random(z.shape) < 0.7
    cfg = PRESETS["cimple"].with_(recip=recip)
    q8, ratio = _np_ref(z, v, keep, 0.04, recip == "lut")
    res = split_softmax_int(torch.from_numpy(z), torch.from_numpy(v), torch.from_numpy(keep), cfg, 0.04)
    assert res["out"].dtype == torch.int64
    # same float64 ops up to summation order: allow ulp-level ratio error, then compare ints
    resf = split_softmax_int(torch.from_numpy(z), torch.from_numpy(v), torch.from_numpy(keep),
                             cfg.with_(quantize_out=False), 0.04)
    np.testing.assert_allclose(resf["out"].numpy(), ratio, rtol=1e-12, atol=1e-12)
    assert np.array_equal(res["out"].numpy(), q8)


def test_hw8_exact_recip_integer_rounding():
    rng = _rng()
    z = rng.integers(-128, 128, size=(4, 6, 30))
    v = rng.integers(-128, 128, size=(4, 30, 3))
    cfg = HW8.with_(recip="exact")
    res = split_softmax_int(torch.from_numpy(z), torch.from_numpy(v), None, cfg, 0.05)
    p, s = res["partial"].numpy(), res["sum"].numpy()[..., None]
    from fractions import Fraction
    exp = np.vectorize(lambda a, b: 0 if b == 0 else
                       int(np.sign(a)) * int(Fraction(abs(int(a)), int(b)) + Fraction(1, 2)))(p, s)
    assert np.array_equal(res["out"].numpy(), np.clip(exp, -128, 127))


def test_float_v_gives_float_ratio():
    rng = _rng()
    z = torch.from_numpy(rng.integers(-128, 128, size=(2, 3, 10)))
    v = torch.randn(2, 10, 4, dtype=torch.float32)
    res = split_softmax_int(z, v, None, PRESETS["cimple"], 0.05)
    assert res["out"].dtype == torch.float64
    e = res["e"].double()
    torch.testing.assert_close(res["out"], (e / e.sum(-1, keepdim=True)) @ v.double())


def test_out_of_range_score_rejected():
    with pytest.raises(ValueError):
        split_softmax_int(torch.tensor([[200]]), torch.tensor([[1]]), None, HW8, 0.05)


# ---------------------------------------------------------------------------
# attention level
# ---------------------------------------------------------------------------
def _qkv(B, H, L, D, score_std, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, L, D, generator=g)
    k = torch.randn(B, H, L, D, generator=g)
    v = torch.randn(B, H, L, D, generator=g)
    q = q * score_std                 # q.k / sqrt(D) has std ~= score_std
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _cos(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def _causal(L):
    return torch.ones(L, L, dtype=torch.bool).tril()


@pytest.mark.parametrize("L", [64, 256])
@pytest.mark.parametrize("std", [2.0, 5.0])
def test_cimple_preset_close_to_float(L, std):
    B, H, D = 2, 4, 64
    q, k, v = _qkv(B, H, L, D, std, seed=L)
    keep = _causal(L)
    ref = float_attention(q, k, v, scaling=D ** -0.5, keep=keep)
    out = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["cimple"])
    cos, mae = _cos(out, ref), (out - ref).abs().mean().item()
    print(f"\n[cimple] L={L} std={std}: cos={cos:.5f} mae={mae:.4f} (|ref| mean "
          f"{ref.abs().mean().item():.3f})")
    assert cos > 0.99
    assert mae < 0.05


@pytest.mark.parametrize("std", [2.0, 5.0])
def test_hw8_presets_degrade_honestly(std):
    B, H, L, D = 2, 4, 128, 64
    q, k, v = _qkv(B, H, L, D, std, seed=7)
    keep = _causal(L)
    ref = float_attention(q, k, v, scaling=D ** -0.5, keep=keep)
    res = {}
    for name in ("cimple_hw8", "cimple_hw8_rowmax"):
        out, aux = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS[name],
                                    return_aux=True)
        res[name] = (_cos(out, ref), (aux["sum"] == 0).float().mean().item())
        print(f"\n[{name}] std={std}: cos={res[name][0]:.4f} "
              f"rows with exp-sum 0 = {100 * res[name][1]:.1f}%")
    # const max + 8b table: rows whose max sits far below z_quant_max underflow to an
    # exp-sum of 0 and output 0 (docs/01-spec-teardown.md "Known weakness"). Measured
    # (seed 7): cos 0.908 / 3.4% dead rows at std 2, cos 0.515 / 79.6% at std 5.
    floor = {2.0: 0.85, 5.0: 0.4}[std]
    assert res["cimple_hw8"][0] > floor
    assert res["cimple_hw8"][0] < res["cimple_hw8_rowmax"][0]
    assert res["cimple_hw8_rowmax"][0] > 0.99 and res["cimple_hw8_rowmax"][1] == 0


def test_asymmetric_scores_absmax_scale_kills_hw8_head():
    """N2, documented [A]: the dynamic scale is symmetric absmax. One strongly negative
    key (score ~ -40 vs a max of ~ +5) pushes every z far below z_quant_max, so the
    hw8 + const-max head underflows completely; a positive-side static scale fixes it."""
    L = D = 64
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 1, L, D, generator=g) for _ in range(3))
    c = torch.randn(D, generator=g)
    q = q + c
    k[..., 0, :] = -3 * c
    keep = _causal(L)
    sc = (q @ k.transpose(-2, -1)) * D ** -0.5
    kmax, kmin = sc.masked_fill(~keep, -1e9).max(), sc.masked_fill(~keep, 1e9).min()
    assert kmax < 10 and kmin < -30                               # asymmetric heads
    ref = float_attention(q, k, v, scaling=D ** -0.5, keep=keep)
    run = lambda cfg, **kw: cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=cfg,
                                             return_aux=True, **kw)
    out, aux = run(PRESETS["cimple_hw8"])
    assert aux["z"].masked_fill(~keep, -128).max() < 30           # max z ~17, not 127
    assert (aux["sum"] == 0).all() and (out == 0).all()           # whole head dead
    out, aux = run(PRESETS["cimple"])                             # full table survives
    assert (aux["sum"] > 0).all() and _cos(out, ref) > 0.99
    out, aux = run(PRESETS["cimple_hw8_rowmax"])
    assert (aux["sum"] > 0).all() and _cos(out, ref) > 0.99
    out, aux = run(PRESETS["cimple_hw8"].with_(score_scale="static"), score_scale=kmax / 127)
    assert (aux["sum"] == 0).sum() <= 2 and _cos(out, ref) > 0.95  # measured 1 dead, cos 0.964


def test_score_rint_vs_rtl_requantiser_bound():
    """N4: float rint(score/s) vs the RTL (scale, shift) requantiser differ by <= 2 LSB on
    ~30% of ordinary inputs -- almost all negative (the -1 LSB bias of gm.quantise),
    the rest from the 16b scale approximation; not just on exact ties."""
    rng = _rng()
    d, neg = [], []
    for _ in range(300):
        r = 10 ** rng.uniform(-4, -2)
        sc, sh = gm.ratio_to_scale_shift(r)
        acc = rng.integers(-200000, 200000, size=50)
        rtl = luts.requantise(torch.from_numpy(acc), sc, sh).numpy()
        assert rtl.tolist() == [gm.quantise(int(a), sc, sh) for a in acc]
        d.append(rtl - np.clip(np.rint(acc * r), -128, 127))
        neg.append(acc < 0)
    d, neg = np.concatenate(d), np.concatenate(neg)
    assert np.abs(d).max() <= 2
    assert 0.2 < (d != 0).mean() < 0.4
    assert (d[~neg] != 0).mean() < 0.01 and (d[neg] != 0).mean() > 0.5 and (d[neg] <= 0).all()


def test_causal_mask_respected():
    B, H, L, D = 1, 2, 48, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=3)
    keep = _causal(L)
    t = 20
    k2, v2 = k.clone(), v.clone()
    k2[..., t + 1:, :] = torch.randn_like(k2[..., t + 1:, :]) * 0.5
    v2[..., t + 1:, :] = torch.randn_like(v2[..., t + 1:, :]) * 0.5
    # static score scale + float q/k/v: no per-head statistic sees the future
    cfg = PRESETS["cimple"].with_(score_scale="static", quantize_qkv=False)
    a = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=cfg, score_scale=0.05)
    b = cimple_attention(q, k2, v2, scaling=D ** -0.5, keep=keep, cfg=cfg, score_scale=0.05)
    assert torch.equal(a[..., : t + 1, :], b[..., : t + 1, :])
    assert not torch.equal(a[..., t + 1:, :], b[..., t + 1:, :])
    # every preset: masked positions get exactly zero weight
    for name in ("cimple", "cimple_hw8", "cimple_hw8_rowmax"):
        _, aux = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS[name],
                                  return_aux=True)
        assert (aux["e"][..., ~keep] == 0).all() and (aux["probs"][..., ~keep] == 0).all()


@pytest.mark.parametrize("name", list(PRESETS))
def test_fully_masked_row_is_zero(name):
    B, H, L, D = 1, 2, 16, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=5)
    keep = _causal(L).clone()
    keep[3] = False
    out = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS[name])
    assert torch.isfinite(out).all()
    assert (out[..., 3, :] == 0).all()


def test_bf16_input():
    B, H, L, D = 1, 2, 32, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=9)
    keep = _causal(L)
    out16 = cimple_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), scaling=D ** -0.5,
                             keep=keep, cfg=PRESETS["cimple"])
    out32 = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["cimple"])
    assert out16.dtype == torch.bfloat16 and out16.shape == (B, H, L, D)
    assert _cos(out16, out32) > 0.99


def test_static_scale_reproduces_dynamic():
    B, H, L, D = 2, 3, 40, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=11)
    keep = _causal(L)
    cfg = PRESETS["cimple_hw8"]
    dyn, aux = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=cfg, return_aux=True)
    st = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=cfg.with_(score_scale="static"),
                          score_scale=aux["score_scale"])
    assert torch.equal(dyn, st)
    with pytest.raises(ValueError):
        cimple_attention(q, k, v, scaling=D ** -0.5, cfg=cfg.with_(score_scale="static"))
    # calibration helper agrees with the dynamic scale up to Q/K rounding
    amax = observe_score_absmax(q, k, scaling=D ** -0.5, keep=keep)
    assert amax.shape == (B, H)
    torch.testing.assert_close(amax.double() / 127, aux["score_scale"][..., 0, 0], rtol=0.05, atol=0)


def test_float_qkv_and_float_mode():
    B, H, L, D = 1, 2, 64, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=13)
    keep = _causal(L)
    ref = float_attention(q, k, v, scaling=D ** -0.5, keep=keep)
    assert torch.equal(cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["float"]), ref)
    fq = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["float"].with_(quantize_qkv=True))
    assert _cos(fq, ref) > 0.999
    nq = cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS["cimple"].with_(quantize_qkv=False))
    assert _cos(nq, ref) > 0.995


def test_aux_contents():
    B, H, L, D = 1, 2, 24, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=17)
    _, aux = cimple_attention(q, k, v, scaling=D ** -0.5, keep=_causal(L), cfg=PRESETS["cimple_hw8"],
                              return_aux=True)
    assert aux["z"].dtype == torch.int64 and aux["z"].shape == (B, H, L, L)
    assert aux["score_scale"].shape == (B, H, 1, 1)
    s = aux["probs"].sum(-1)
    assert torch.allclose(s[aux["sum"] > 0], torch.ones(()), atol=1e-5)


def test_speed():
    B, H, L, D = 1, 32, 1024, 64
    q, k, v = _qkv(B, H, L, D, 3.0, seed=19)
    keep = _causal(L)
    for name in ("cimple", "cimple_hw8"):
        cimple_attention(q[..., :8, :], k, v, scaling=D ** -0.5, keep=keep[:8], cfg=PRESETS[name])  # warm-up
        t0 = time.perf_counter()
        cimple_attention(q, k, v, scaling=D ** -0.5, keep=keep, cfg=PRESETS[name])
        dt = time.perf_counter() - t0
        print(f"\n[speed] {name} B=1 H=32 L=1024 D=64: {dt:.2f} s")
        assert dt < 10.0
