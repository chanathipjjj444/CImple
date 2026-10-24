"""Tests for the CIMple mapping simulator (cimple_sw.mapping).

The references below are plain numpy integer maths -- matmuls, the golden
quantiser and the golden LUT definitions -- with no tiling, banks or passes,
so equality with the simulator checks the hardware schedule end to end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cimple_rtl.model import cimple_model as gm
from cimple_sw import mapping as mp
from cimple_sw.hw import Timing

gq = np.vectorize(gm.quantise, otypes=[np.int64])   # the golden scalar quantiser


# ---------------------------------------------------------------------------
# Independent integer reference
# ---------------------------------------------------------------------------
def ref_block(xq, xkv, w: mp.AttnWeights, sc: mp.Scales, causal=False, e_mode="shift1", zq=127):
    """softmax-split attention block, plain integer maths, golden quantiser."""
    H = w.n_heads
    dh = w.wq.shape[1] // H
    Wq, Wk, Wv = (gm.to_int8(W, s) for W, s in ((w.wq, sc.wq), (w.wk, sc.wk), (w.wv, sc.wv)))
    Q = gq(xq @ Wq, *gm.ratio_to_scale_shift(sc.x * sc.wq / sc.q))
    K = gq(xkv @ Wk, *gm.ratio_to_scale_shift(sc.x_kv * sc.wk / sc.k))
    V = gq(xkv @ Wv, *gm.ratio_to_scale_shift(sc.x_kv * sc.wv / sc.v))
    recip = gm.build_recip_lut()
    heads = []
    for h in range(H):
        q, k, v = (T[:, h * dh:(h + 1) * dh] for T in (Q, K, V))
        z = gq(q @ k.T, *gm.ratio_to_scale_shift(sc.q * sc.k / math.sqrt(dh) / sc.score[h]))
        e = gm.build_exp_lut(sc.score[h], zq)[z % 256]
        if causal:
            for i in range(len(q)):
                e[i, i + 1:] = 0
        a = e // 2 if e_mode == "shift1" else e
        out = np.zeros((len(q), dh), dtype=np.int64)
        for i in range(len(q)):
            s = int(a[i].sum())
            M = int(recip[gm.mantissa8(s)])
            sh = gm.RECIP_FRAC - gm.LUT_W + gm.msb_pos(s)
            out[i] = gq(a[i] @ v, M, sh)
        heads.append(out)
    cat = np.concatenate(heads, axis=1)
    if w.wo is None:
        return cat
    Wo = gm.to_int8(w.wo, sc.wo)
    return gq(cat @ Wo, *gm.ratio_to_scale_shift(sc.v * sc.wo / sc.o))


def float_mha(X, w: mp.AttnWeights, Xkv=None, causal=False):
    Xkv = X if Xkv is None else Xkv
    H, dh = w.n_heads, w.d_head
    Q, K, V = X @ w.wq, Xkv @ w.wk, Xkv @ w.wv
    outs = []
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        s = Q[:, sl] @ K[:, sl].T / np.sqrt(dh)
        if causal:
            s = s + np.triu(np.full(s.shape, -np.inf), 1)
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        outs.append(p @ V[:, sl])
    cat = np.concatenate(outs, axis=1)
    return cat if w.wo is None else cat @ w.wo


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# 1. the vectorised core
# ---------------------------------------------------------------------------
def test_core_pass_matches_golden(rng):
    W = rng.integers(-128, 128, (32, 64))
    A = rng.integers(-128, 128, (6, 64))
    corners = [np.full((32, 64), v) for v in (-128, 127)]
    acts = [np.full(64, v) for v in (-128, 127, 0, -1)]
    cases = [(W, a) for a in A] + [(c, a) for c in corners for a in acts]
    for Wc, a in cases:
        assert np.array_equal(mp.core_pass(Wc, a[None])[0], gm.core_mac(Wc, a))
    # batched == one at a time
    assert np.array_equal(mp.core_pass(W, A), np.stack([gm.core_mac(W, a) for a in A]))


def test_core_pass_unsigned_mode(rng):
    W = rng.integers(-128, 128, (32, 64))
    A = np.concatenate([rng.integers(0, 256, (5, 64)), np.full((1, 64), 255)])
    for Wc in (W, np.full((32, 64), -128), np.full((32, 64), 127)):
        assert np.array_equal(mp.core_pass(Wc, A, unsigned=True), A @ Wc.T)


def test_quantise_vec_matches_golden(rng):
    acc = np.concatenate([rng.integers(-(1 << 31), 1 << 31, 2000), [0, 1, -1, 3, -3, 5, -5]])
    for sc, sh in [(1, 0), (3, 1), (40000, 20), (12345, 31), (128, 7)]:
        assert np.array_equal(mp.quantise_vec(acc, sc, sh), gq(acc, sc, sh))


def test_core_bank_rules():
    core = mp.CimpleCore()
    tile = np.ones((32, 64), dtype=np.int64)
    core.load_next(tile)
    assert core.stats.stages["proj"].write == 32 * 4          # 128b port: 16 weights/cycle
    core.mac(np.ones((1, 64)), used=(32, 64))
    with pytest.raises(AssertionError, match="active bank"):
        core.write(core.active, tile)
    core.write(core.active ^ 1, tile)                         # shadow bank is fine
    # the shadow write (128 cycles) overlaps nothing -> the next pass stalls
    core.select(core.active ^ 1)
    core.mac(np.ones((1, 64)), used=(32, 64))
    st = core.stats.stages["proj"]
    assert st.passes == 2 and st.core == 2 * Timing().pass_cycles and st.stall > 0


# ---------------------------------------------------------------------------
# 2. each mapping == the integer reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N,dm,H,dh", [(48, 128, 4, 32), (70, 96, 2, 48), (20, 64, 1, 64)])
def test_encoder_matches_reference(rng, N, dm, H, dh):
    w = mp.AttnWeights.random(dm, H, dh, rng)
    X = rng.standard_normal((N, dm))
    r = mp.encoder_layer(X, w)
    xq = gm.to_int8(X, r.scales.x)
    assert np.array_equal(r.out, ref_block(xq, xq, w, r.scales))


def test_encoder_causal_and_no_wo_match_reference(rng):
    w = mp.AttnWeights.random(64, 2, 32, rng, out_proj=False)
    X = rng.standard_normal((100, 64))
    r = mp.encoder_layer(X, w, causal=True)
    xq = gm.to_int8(X, r.scales.x)
    assert np.array_equal(r.out, ref_block(xq, xq, w, r.scales, causal=True))
    assert r.out_scale == r.scales.v


def test_decoder_matches_reference(rng):
    w = mp.AttnWeights.random(64, 2, 32, rng)
    X = rng.standard_normal((40, 64))
    r = mp.decoder_layer(X, w)
    xq = gm.to_int8(X, r.scales.x)
    assert np.array_equal(r.out, ref_block(xq, xq, w, r.scales, causal=True))


# ---------------------------------------------------------------------------
# 3. decoder token-by-token == encoder with causal mask, same static scales
# ---------------------------------------------------------------------------
def test_decoder_equals_causal_encoder(rng):
    w = mp.AttnWeights.random(96, 3, 32, rng)
    X = rng.standard_normal((72, 96))
    enc = mp.encoder_layer(X, w, causal=True)
    dec = mp.decoder_layer(X, w, enc.scales)
    assert np.array_equal(dec.out, enc.out)
    assert np.array_equal(dec.k, enc.k) and np.array_equal(dec.v, enc.v)
    # prefill on the encoder dataflow, then continue token by token
    pre = mp.encoder_layer(X[:65], w, enc.scales, causal=True)
    cont = mp.decoder_layer(X[65:], w, enc.scales, cache=pre.cache)
    assert np.array_equal(cont.out, enc.out[65:])
    assert cont.stats.tokens == 7


# ---------------------------------------------------------------------------
# 4. encoder-decoder
# ---------------------------------------------------------------------------
def test_encdec_matches_reference(rng):
    dm, H, dh = 64, 2, 32
    ew, sw, cw = (mp.AttnWeights.random(dm, H, dh, rng) for _ in range(3))
    Xe = rng.standard_normal((80, dm))
    Xd = rng.standard_normal((10, dm))
    r = mp.encdec_layer(Xe, Xd, ew, sw, cw)
    s = r.scales
    xe = gm.to_int8(Xe, s["enc"].x)
    E = ref_block(xe, xe, ew, s["enc"])
    assert np.array_equal(r.enc.out, E)
    xd = gm.to_int8(Xd, s["self"].x)
    Y1 = ref_block(xd, xd, sw, s["self"], causal=True)
    assert np.array_equal(r.self_out, Y1)
    Y2 = ref_block(Y1, E, cw, s["cross"])            # cross: Q from Y1, K/V from E
    assert np.array_equal(r.out, Y2)
    # dequantised cross output tracks the float encoder-decoder
    Ef = float_mha(Xe, ew)
    Y1f = float_mha(Xd, sw, causal=True)
    c = cosine(r.out * r.out_scale, float_mha(Y1f, cw, Ef))
    print(f"\nenc-dec cross-attention cosine vs float: {c:.4f}")
    assert c > 0.9


# ---------------------------------------------------------------------------
# 5. accuracy vs float attention
# ---------------------------------------------------------------------------
def test_accuracy_vs_float(rng):
    cs = []
    for _ in range(4):
        w = mp.AttnWeights.random(128, 4, 32, rng)
        X = rng.standard_normal((48, 128))
        r = mp.encoder_layer(X, w)
        cs.append(cosine(r.dequant(), float_mha(X, w)))
    print(f"\nencoder N=48 dm=128 H=4 dh=32, cosine vs float64: "
          f"min {min(cs):.4f} mean {np.mean(cs):.4f}")
    assert min(cs) > 0.95


# ---------------------------------------------------------------------------
# 6. e-activation options and stats sanity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("e_mode", ["shift1", "unsigned"])
def test_e_modes_match_reference(rng, e_mode):
    w = mp.AttnWeights.random(64, 2, 32, rng)
    X = rng.standard_normal((70, 64))
    r = mp.encoder_layer(X, w, e_mode=e_mode)
    xq = gm.to_int8(X, r.scales.x)
    assert np.array_equal(r.out, ref_block(xq, xq, w, r.scales, e_mode=e_mode))
    d = mp.decoder_layer(X[:20], w, r.scales, e_mode=e_mode)
    assert np.array_equal(d.out, ref_block(xq[:20], xq[:20], w, r.scales, causal=True,
                                           e_mode=e_mode))


def test_e_modes_vs_ideal_softmax():
    """Both modes against the exact softmax of the same INT8 scores and V.

    Measured finding: e >> 1 throws away the LSB of small exp values, which
    with the constant z_quant_max are common; on rows whose exp values are all
    small (e = 1..3) the output collapses onto a few keys. The unsigned mode
    keeps that bit and is consistently closer.
    """
    rng = np.random.default_rng(0)
    err = {"shift1": [], "unsigned": []}
    for _ in range(3):
        w = mp.AttnWeights.random(64, 2, 32, rng, out_proj=False)
        X = rng.standard_normal((64, 64))
        a = mp.encoder_layer(X, w, e_mode="shift1")
        b = mp.encoder_layer(X, w, a.scales, e_mode="unsigned")
        sc, dh, ideal = a.scales, 32, []
        for h in range(2):
            sl = slice(h * dh, (h + 1) * dh)
            z = gq(a.q[:, sl] @ a.k[:, sl].T, *gm.ratio_to_scale_shift(sc.score_ratio(h, dh)))
            p = np.exp((z - z.max(1, keepdims=True)) * sc.score[h])
            ideal.append((p / p.sum(1, keepdims=True)) @ a.v[:, sl])
        ideal = np.concatenate(ideal, axis=1)
        err["shift1"].append(np.abs(a.attn - ideal).mean())
        err["unsigned"].append(np.abs(b.attn - ideal).mean())
    print(f"\nmean |err| vs ideal softmax (INT8 LSB): shift1 {np.round(err['shift1'], 3)}, "
          f"unsigned {np.round(err['unsigned'], 3)}")
    assert all(u < s for u, s in zip(err["unsigned"], err["shift1"]))
    assert max(err["unsigned"]) < 1.5


def test_encoder_stats_analytic(rng):
    N, dm, H, dh = 80, 128, 4, 32
    w = mp.AttnWeights.random(dm, H, dh, rng)
    r = mp.encoder_layer(rng.standard_normal((N, dm)), w)
    st, hd = r.stats, H * dh
    assert st.stages["proj"].macs == 3 * N * dm * hd
    assert st.stages["qkt"].macs == H * N * N * dh
    assert st.stages["av"].macs == H * N * N * dh
    assert st.stages["out_proj"].macs == N * hd * dm
    assert st.ops == 2 * st.macs
    # passes: projection = tokens x tiles; QK^T = keys per query tile; A'V = queries per chunk
    tiles = lambda K, M: math.ceil(K / 64) * math.ceil(M / 32)  # noqa: E731
    assert st.stages["proj"].passes == 3 * N * tiles(dm, hd)
    assert st.stages["qkt"].passes == H * math.ceil(N / 32) * N
    assert st.stages["av"].passes == H * math.ceil(N / 64) * N
    pc = Timing().pass_cycles
    for s in st.stages.values():
        assert s.core == s.passes * pc
    assert 0 < st.utilisation <= 1
    assert st.cycles >= sum(s.core for s in st.stages.values())


def test_decoder_step_stats_analytic(rng):
    dm, H, dh, L = 128, 4, 32, 100
    w = mp.AttnWeights.random(dm, H, dh, rng)
    X = rng.standard_normal((L, dm))
    enc = mp.encoder_layer(X, w, causal=True)
    pre = mp.encoder_layer(X[:L - 1], w, enc.scales, causal=True)
    d = mp.decoder_layer(X[L - 1:], w, enc.scales, cache=pre.cache)
    assert np.array_equal(d.out[0], enc.out[-1])
    st, hd = d.stats, H * dh
    assert st.stages["proj"].macs == 3 * dm * hd
    assert st.stages["qkt"].macs == H * L * dh
    assert st.stages["av"].macs == H * L * dh
    assert st.stages["qkt"].passes == H * math.ceil(L / 32)
    # every K and V element of the context is written into the CIM once per head
    assert st.stages["qkt"].write == H * L * math.ceil(dh / 16)
    assert st.stages["av"].write == H * sum(dh * math.ceil(min(64, L - c) / 16)
                                            for c in range(0, L, 64))
    assert st.ops == 2 * st.macs and st.tokens == 1


def test_overlap_out_timing_is_faster(rng):
    w = mp.AttnWeights.random(64, 2, 32, rng)
    X = rng.standard_normal((64, 64))
    a = mp.encoder_layer(X, w)
    b = mp.encoder_layer(X, w, a.scales, timing=Timing(overlap_out=True))
    assert np.array_equal(a.out, b.out)
    assert b.stats.cycles < a.stats.cycles


def test_main_runs(capsys):
    assert mp.main(["--n", "40", "--d-model", "64", "--heads", "2", "--dec-tokens", "3"]) == 0
    out = capsys.readouterr().out
    assert "encoder-only" in out and "decode step" in out and "encoder-decoder" in out


# ---------------------------------------------------------------------------
# 7. schedule / buffer / underflow regressions
# ---------------------------------------------------------------------------
def test_encoder_dh64_hides_v_writes(rng):
    """Head dim 64 (two V tiles per chunk): with ping-pong placement every
    bank write hides behind a pass, as in latency.py's default schedule."""
    from cimple_sw import latency as lt

    N = 256
    w = mp.AttnWeights.random(64, 1, 64, rng, out_proj=False)
    X = rng.standard_normal((N, 64))
    r = mp.encoder_layer(X, w)
    xq = gm.to_int8(X, r.scales.x)
    assert np.array_equal(r.out, ref_block(xq, xq, w, r.scales))
    st = r.stats.stages
    assert st["av"].stall == 0
    assert st["qkt"].stall <= 32 * Timing.write_cycles(64)     # only the very first Q tile
    attn = r.stats.phases["attn"]
    assert attn == st["qkt"].core + st["av"].core + st["qkt"].stall
    ref = lt.latency(N, 64).cycles
    assert abs(attn - ref) / ref < 0.01


def test_e_buffer_peak(rng):
    w = mp.AttnWeights.random(64, 2, 32, rng)
    X = rng.standard_normal((100, 64))
    enc = mp.encoder_layer(X, w)
    assert enc.stats.buf_e_max == 32 * 64                  # 32 queries x 64 keys corner turn
    dec = mp.decoder_layer(X[:70], w, enc.scales)
    assert dec.stats.buf_e_max == 64


@pytest.mark.parametrize("gain", [1.0, 2.0])
def test_exp_underflow_zero_rows(gain):
    """Constant z_quant_max + per-head absmax score scale: the 8b exp LUT
    underflows on larger inputs, whole rows get exp-sum 0 and are output as 0.
    The accuracy of test_accuracy_vs_float holds only for unit-variance inputs."""
    rng = np.random.default_rng(1234)
    H, dh, N = 4, 32, 48
    w = mp.AttnWeights.random(128, H, dh, rng)
    X = gain * rng.standard_normal((N, 128))
    r = mp.encoder_layer(X, w)
    zero_out = sum(int(np.all(r.attn[:, h * dh:(h + 1) * dh] == 0, axis=1).sum())
                   for h in range(H))
    assert r.stats.zero_rows <= zero_out
    c = cosine(r.dequant(), float_mha(X, w))
    frac = r.stats.zero_rows / (N * H)
    print(f"\ngain {gain}: zero rows {frac:.0%}, cosine vs float {c:.3f}")
    if gain == 1.0:
        assert r.stats.zero_rows == 0 and c > 0.95
    else:
        assert frac > 0.5 and c < 0.6
        assert "exp-sum 0" in r.stats.table()
