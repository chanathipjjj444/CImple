"""Model-level proof for the near-memory (VP1202) schedule.

Same idea as test_attention_model.py, one level up: before any RTL is written
for the sequencer, show that the tiled, double-buffered schedule computes the
same integers as plain matrix maths, that it never corrupts the active bank,
and what it costs in cycles.

Run:  uv run pytest tests/test_nearmem_model.py -s
"""
import numpy as np
import pytest

from cimple_rtl.model import cimple_model as m
from cimple_rtl.model import nearmem_model as nm


def rand8(rng, shape):
    return rng.integers(-128, 128, shape)


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- the building block ------------------------------------------------------

def test_core_pass_matches_partition_mac():
    """The vectorised pass is the same bit-serial algorithm as the model's."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        W = rand8(rng, (m.N_PART, m.N_ROWS))
        x = rand8(rng, m.N_ROWS)
        assert np.array_equal(nm.core_pass(W, x), m.core_mac(W, x))
    corners = np.array([-128, 127])
    for w in corners:
        for a in corners:
            W = np.full((m.N_PART, m.N_ROWS), w)
            x = np.full(m.N_ROWS, a)
            assert np.array_equal(nm.core_pass(W, x), m.core_mac(W, x))


# --- exactness of the tiled schedule ---------------------------------------

@pytest.mark.parametrize("N,K,M", [
    (8, 64, 32),        # exactly one tile, one block
    (13, 100, 50),      # nothing a multiple of anything: padding everywhere
    (40, 256, 96),      # 4 K tiles, 3 M tiles, 5 blocks
    (128, 256, 256),    # BERT-mini projection
])
def test_tiled_matmul_is_exact(N, K, M):
    rng = np.random.default_rng(N * K + M)
    A, W = rand8(rng, (N, K)), rand8(rng, (K, M))
    Y, st = nm.nm_matmul(A, W)
    assert np.array_equal(Y, A @ W)
    assert st.macs == N * K * M
    print(f"\n{N}x{K} @ {K}x{M}: " + nm.summary(st))


def test_stale_shadow_bank_is_harmless():
    """Decision 3: padding relies on zero activations, not zeroed storage.
    Different power-up garbage must give identical results."""
    rng = np.random.default_rng(5)
    A, W = rand8(rng, (11, 70)), rand8(rng, (70, 33))
    outs = [nm.nm_matmul(A, W, seed=s)[0] for s in range(4)]
    for Y in outs:
        assert np.array_equal(Y, A @ W)


def test_loader_never_writes_active_bank():
    """Decision 1, checked directly on the Array, not just via the result."""
    arr = nm.Array(np.random.default_rng(0))
    tile = np.zeros((m.N_PART, m.N_ROWS), dtype=np.int64)
    arr.load(arr.active ^ 1, "ok", tile)
    with pytest.raises(AssertionError, match="active bank"):
        arr.load(arr.active, "bad", tile)


# --- the schedule: does double buffering hide the reload? ------------------

def test_reload_is_hidden_at_128b_write_bus():
    """With the paper's 128b write bus, a full tile loads in 2048/16 = 128
    cycles, and a full 8-token block computes in 8 x 16 = 128 cycles. They
    are exactly equal at DEPTH = 64, so a full-size layer never stalls."""
    rng = np.random.default_rng(1)
    A, W = rand8(rng, (128, 256)), rand8(rng, (256, 256))
    _, st = nm.nm_matmul(A, W)
    assert st.stall == 0
    assert st.load_hidden == (st.loads - 1) * 128


def test_reload_stalls_with_todays_8b_write_port():
    """The current RTL writes one weight per cycle: 2048-cycle loads against
    128-cycle blocks. Still exact, but the core idles ~94 % of the time."""
    rng = np.random.default_rng(2)
    A, W = rand8(rng, (64, 256)), rand8(rng, (256, 64))
    cfg = nm.Config(load_wpc=1)
    Y, st = nm.nm_matmul(A, W, cfg)
    assert np.array_equal(Y, A @ W)
    assert st.stall > 10 * st.compute
    print(f"\n8b write port: " + nm.summary(st, cfg))


def test_bert_mini_fits_in_uram():
    """BERT-mini encoder weights (4 layers, d=256, ffn=1024) in INT8."""
    per_layer = 4 * 256 * 256 + 2 * 256 * 1024
    bits = 4 * per_layer * m.DATA_W
    print(f"\nBERT-mini encoder: {bits / 2**20:.1f} Mb of {nm.URAM_BITS / 2**20:.0f} Mb UltraRAM")
    assert bits < nm.URAM_BITS


# --- end to end: an attention head --------------------------------------------

def make_head(seed, n, dm, d, w_std):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, (n, dm))
    return (X,) + tuple(rng.normal(0, w_std, (dm, d)) for _ in range(3))


def test_matches_single_tile_model_where_it_applies():
    """Inside the old single-tile limits, stages 1-2 must agree bit for bit
    with attention_head_cim (stage 3-4 differ by decision 4: e >> 1)."""
    X, Wq, Wk, Wv = make_head(0, 16, 64, 32, 0.15)
    ref = m.attention_head_cim(X, Wq, Wk, Wv)
    got = nm.attention_head_nm(X, Wq, Wk, Wv)
    for k in ("Q", "K", "V", "z"):
        assert np.array_equal(got[k], ref[k]), k


def test_bert_mini_head_tracks_float():
    """One BERT-mini-shaped head: 128 tokens, d_model 256, d_head 64.
    Synthetic Gaussian data, not trained weights."""
    cos = []
    for seed in range(3):
        X, Wq, Wk, Wv = make_head(seed, 128, 256, 64, 1 / np.sqrt(256))
        ref, _ = m.attention_head_float(X, Wq, Wk, Wv)
        r = nm.attention_head_nm(X, Wq, Wk, Wv)
        got = r["out"] * r["out_scale"]
        cos.append(cosine(ref, got))
        dead = int((r["e7"].sum(axis=1) == 0).sum())
        print(f"\nseed {seed}: cosine {cos[-1]:.4f}, "
              f"rel L2 {np.linalg.norm(ref - got) / np.linalg.norm(ref):.4f}, "
              f"dead rows {dead}/128")
        if seed == 0:
            total = nm.Stats()
            for name, st in r["stats"].items():
                print(f"  {name:7s} " + nm.summary(st))
                total.merge(st)
            print(f"  head    " + nm.summary(total))
            print(f"  (DSP58 peak on VP1202: {nm.DSP_PEAK_INT8_TOPS} INT8 TOPS)")
    assert min(cos) > 0.9, f"worst cosine {min(cos):.3f}"
