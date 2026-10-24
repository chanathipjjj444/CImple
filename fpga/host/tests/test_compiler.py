"""Compiler: programs on the ISA simulator == plain integer references (exact),
close to float attention, and bit-exact with cimple_sw.mapping (e_mode="unsigned").

Cosine thresholds: measured over seeds 100..105 (fpga/host, unit-variance X,
W ~ N(0, 1/d_model)): encoder N=64 d=64 min 0.986; N=256 d=64 min 0.973;
N=256 d=32 causal min 0.916; N=128 dm=128 d=64 causal min 0.990; decoder step
(one row, L=101) min 0.948; cross (20 x 150) min 0.981. The asserted bounds sit
just below the values measured on the fixed seeds used here (e.g. N=256 d=64
gives 0.9696 with seed 100, bound 0.96).
"""
import numpy as np
import pytest

from cimple_rtl.model import cimple_model as gm
from cpl import compiler as C
from cpl import isa
from cpl.isa import Op, Out


def rint8(rng, *shape):
    return rng.integers(-128, 128, shape)


# ---------------------------------------------------------------------------
# planner / layout
# ---------------------------------------------------------------------------
def test_planner_alignment_and_overflow():
    p = C.Planner(4096)
    a = p.alloc("a", 3, 5)
    b = p.act("b", 2, 70)
    c = p.alloc("c", 1, 4, elem=4)
    assert a.addr == 0 and a.pitch == 16
    assert b.addr == 48 and b.pitch == 128 and c.addr % 16 == 0
    with pytest.raises(MemoryError):
        p.alloc("d", 100, 64)
    r = C.Region("r", 0, 2, 3, 16, elem=4)
    v = np.array([[1, -2, 3], [-(1 << 31), (1 << 31) - 1, 0]])
    assert np.array_equal(r.unpack(r.pack(v)), v)


# ---------------------------------------------------------------------------
# matmul (weight projection)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N,K,M", [(1, 1, 1), (7, 64, 32), (40, 65, 33), (130, 150, 70),
                                   (300, 64, 96)])
def test_matmul_exact(N, K, M):
    rng = np.random.default_rng(N * 1000 + K * 10 + M)
    X, W = rint8(rng, N, K), rint8(rng, K, M)
    b = C.matmul(X, W, ratio=1 / (40.0 * K))
    out, st, _ = C.run_sim(b)
    assert np.array_equal(out["Y"], b.ref["Y"])
    assert np.array_equal(b.ref["Y"], C.ref_matmul(X, W, gm.ratio_to_scale_shift(1 / (40.0 * K))))
    # the result is not trivially saturated / zero
    assert 0 < np.mean(np.abs(out["Y"])) < 120
    assert st["passes"] == sum(i.ntok for i in b.program if i.opcode == Op.MATVEC)


def test_matmul_raw32_and_acc_blocks():
    rng = np.random.default_rng(1)
    X, W = rint8(rng, 50, 200), rint8(rng, 200, 40)
    b = C.matmul(X, W, raw32=True, acc_depth=64)             # 8 tokens per ACC block
    out, _, _ = C.run_sim(b, acc_depth=64)
    assert np.array_equal(out["Y"], X @ W)
    b = C.matmul(X, W, 3, 14, acc_depth=64)
    out, _, _ = C.run_sim(b, acc_depth=64)
    assert np.array_equal(out["Y"], b.ref["Y"])
    assert any(i.acc_mode == isa.Acc.ADD for i in b.program)


def test_matmul_extremes():
    X = np.full((5, 64), -128)
    W = np.full((64, 3), -128)
    b = C.matmul(X, W, raw32=True)
    out, _, _ = C.run_sim(b)
    assert np.all(out["Y"] == 64 * 128 * 128)                 # 2^20: full 23b range use
    b = C.matmul(X, W, 1, 0)
    assert np.all(C.run_sim(b)[0]["Y"] == 127)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def _float_setup(rng, N, dm, d, Nkv=None):
    X = rng.standard_normal((N, dm))
    Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
    Xkv = None if Nkv is None else rng.standard_normal((Nkv, dm))
    return X, Ws, Xkv


@pytest.mark.parametrize("N,dm,d,causal,cos_min", [
    (64, 64, 64, False, 0.98), (256, 64, 64, False, 0.96), (256, 64, 32, True, 0.90),
    (128, 128, 64, True, 0.985), (77, 48, 48, True, 0.95), (33, 64, 16, False, 0.95)])
def test_encoder_head(N, dm, d, causal, cos_min):
    rng = np.random.default_rng(100)
    X, Ws, _ = _float_setup(rng, N, dm, d)
    cal = C.calibrate(X, *Ws, causal=causal)
    b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, causal=causal)
    out, st, _ = C.run_sim(b)
    for k in ("Q", "K", "VT", "O"):
        assert np.array_equal(out[k], b.ref[k]), k
    cos = C.cosine(out["O"] * cal.scales.out_scale, C.float_attention(X, *Ws, causal=causal))
    assert cos > cos_min, cos
    assert len(b.segments()) == 1                            # N <= 256 fits one IMEM load


def test_encoder_reference_is_split_softmax():
    """ref_attention row == gm.softmax_row (with e as UQ8, no >>1)."""
    rng = np.random.default_rng(2)
    X, Ws, _ = _float_setup(rng, 40, 64, 32)
    cal = C.calibrate(X, *Ws)
    b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales)
    r = b.ref
    recip = gm.build_recip_lut()
    for i in (0, 17, 39):
        for c in (0, 31):
            row = gm.softmax_row(r["z"][i], r["V"][:, c], cal.scales.exp_lut, recip)
            assert row["score"] == r["O"][i, c] and row["sum"] == r["den"][i]


def test_segments_split_and_unsplittable():
    rng = np.random.default_rng(3)
    X, Ws, _ = _float_setup(rng, 96, 64, 64)
    cal = C.calibrate(X, *Ws, causal=True)
    b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, causal=True)
    segs = b.segments(12)
    assert len(segs) > 3 and all(len(s) <= 13 and s[-1].opcode == Op.END for s in segs)
    assert sum(len(s) - 1 for s in segs) == len(b.program)
    for s in segs:                       # no segment ends with live sums
        exp = [i for i, x in enumerate(s) if x.opcode == Op.MATVEC and x.out_mode == Out.EXP]
        norm = [i for i, x in enumerate(s) if x.opcode == Op.MATVEC and x.out_mode == Out.NORM]
        if exp:
            assert norm and norm[-1] > exp[-1]
    out, st, _ = C.run_sim(b, max_len=12)
    assert np.array_equal(out["O"], b.ref["O"]) and st["runs"] == len(segs)
    with pytest.raises(ValueError):
        b.segments(5)


@pytest.mark.parametrize("L,nt,d", [(100, 1, 64), (0, 1, 32), (31, 1, 64), (64, 5, 64),
                                    (200, 40, 32)])
def test_decoder_step(L, nt, d):
    rng = np.random.default_rng(L + nt)
    N = L + nt
    X, Ws, _ = _float_setup(rng, N, 64, d)
    cal = C.calibrate(X, *Ws, causal=True)
    kc = C.ref_matmul(cal.xq[:L], cal.wk, cal.scales.k)
    vc = C.ref_matmul(cal.xq[:L], cal.wv, cal.scales.v)
    b = C.decoder_attention_step(cal.xq[L:], cal.wq, cal.wk, cal.wv, kc, vc, cal.scales)
    out, st, _ = C.run_sim(b)
    for k in ("Q", "K", "VT", "O"):
        assert np.array_equal(out[k], b.ref[k]), k
    # appended caches = projections of the whole sequence
    assert np.array_equal(out["K"], C.ref_matmul(cal.xq, cal.wk, cal.scales.k))
    # incremental decoding == the causal encoder rows of the same tokens
    enc = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, causal=True)
    assert np.array_equal(out["O"], enc.ref["O"][L:])
    fl = C.float_attention(X, *Ws, causal=True)[L:]
    assert C.cosine(out["O"] * cal.scales.out_scale, fl) > 0.93


@pytest.mark.parametrize("Nd,Ne,d", [(20, 150, 64), (1, 64, 32), (45, 33, 64)])
def test_cross_attention(Nd, Ne, d):
    rng = np.random.default_rng(Nd * Ne)
    Xd, Ws, He = _float_setup(rng, Nd, 64, d, Nkv=Ne)
    cal = C.calibrate(Xd, *Ws, X_kv=He)
    b = C.cross_attention(cal.xq, cal.xkv, cal.wq, cal.wk, cal.wv, cal.scales)
    out, _, _ = C.run_sim(b)
    for k in ("Q", "K", "VT", "O"):
        assert np.array_equal(out[k], b.ref[k]), k
    fl = C.float_attention(Xd, *Ws, X_kv=He)
    assert C.cosine(out["O"] * cal.scales.out_scale, fl) > 0.96


def test_slow_path_same_result():
    rng = np.random.default_rng(4)
    X, Ws, _ = _float_setup(rng, 70, 64, 64)
    cal = C.calibrate(X, *Ws)
    b = C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales)
    out, st1, _ = C.run_sim(b, fast=False)
    assert np.array_equal(out["O"], b.ref["O"])


def test_gbuf_too_small():
    rng = np.random.default_rng(5)
    with pytest.raises(MemoryError):
        C.matmul(rint8(rng, 256, 256), rint8(rng, 256, 64), 1, 10, gbuf_bytes=64 * 1024)


# ---------------------------------------------------------------------------
# bit-exact cross-check with the mapping simulator (sw/src/cimple_sw/mapping.py)
# ---------------------------------------------------------------------------
mp = pytest.importorskip("cimple_sw.mapping")


def _scales_from_mapping(sc, d):
    return C.AttnScales(*(gm.ratio_to_scale_shift(sc.proj_ratio(k)) for k in "qkv"),
                        gm.ratio_to_scale_shift(sc.score_ratio(0, d)),
                        gm.build_exp_lut(sc.score[0]), out_scale=sc.v)


def _int_w(w, sc):
    return gm.to_int8(w.wq, sc.wq), gm.to_int8(w.wk, sc.wk), gm.to_int8(w.wv, sc.wv)


@pytest.mark.parametrize("N,dm,d,causal", [(64, 64, 64, False), (128, 96, 32, True),
                                           (200, 64, 64, False), (90, 64, 64, True)])
def test_mapping_encoder_bit_exact(N, dm, d, causal):
    rng = np.random.default_rng(N + d)
    w = mp.AttnWeights.random(dm, 1, d, rng, out_proj=False)
    X = rng.standard_normal((N, dm))
    sx = mp.input_scale(X)
    sc = mp.calibrate(gm.to_int8(X, sx), sx, w, causal=causal, e_mode="unsigned")
    res = mp.encoder_layer(X, w, sc, causal=causal, e_mode="unsigned")
    b = C.encoder_attention_head(gm.to_int8(X, sc.x), *_int_w(w, sc), _scales_from_mapping(sc, d),
                                 causal=causal)
    out, _, _ = C.run_sim(b)
    assert np.array_equal(out["Q"][:, :d], res.q) and np.array_equal(out["K"], res.k)
    assert np.array_equal(out["O"], res.out)
    assert res.stats.zero_rows == 0


def test_mapping_decoder_bit_exact():
    rng = np.random.default_rng(9)
    N, dm, d = 70, 64, 64
    w = mp.AttnWeights.random(dm, 1, d, rng, out_proj=False)
    X = rng.standard_normal((N, dm))
    sx = mp.input_scale(X)
    sc = mp.calibrate(gm.to_int8(X, sx), sx, w, causal=True, e_mode="unsigned")
    pre = mp.encoder_layer(X[:N - 4], w, sc, causal=True, e_mode="unsigned")
    dec = mp.decoder_layer(X[N - 4:], w, sc, cache=pre.cache, e_mode="unsigned")
    b = C.decoder_attention_step(gm.to_int8(X[N - 4:], sc.x), *_int_w(w, sc), pre.cache.k,
                                 pre.cache.v, _scales_from_mapping(sc, d))
    out, _, _ = C.run_sim(b)
    assert np.array_equal(out["O"], dec.out)
    assert np.array_equal(out["K"], dec.cache.k) and np.array_equal(out["VT"], dec.cache.v.T)


def test_mapping_cross_bit_exact():
    """Cross attention == mapping's plain-integer block with xkv = encoder output."""
    rng = np.random.default_rng(11)
    dm, d, Nd, Ne = 64, 64, 12, 100
    w = mp.AttnWeights.random(dm, 1, d, rng, out_proj=False)
    E = rng.integers(-128, 128, (Ne, dm))
    Y = rng.integers(-128, 128, (Nd, dm))
    sc = mp.calibrate(Y, 0.02, w, E, 0.03, e_mode="unsigned")
    ref = mp._functional_block(Y, E, w, sc, False, "unsigned", 127)
    b = C.cross_attention(Y, E, *_int_w(w, sc), _scales_from_mapping(sc, d))
    out, _, _ = C.run_sim(b)
    assert np.array_equal(out["O"], ref)
