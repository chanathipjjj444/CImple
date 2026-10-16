"""Model-level proof: the CIMple datapath computes attention like a DNN layer.

No RTL here. This is the target the RTL must hit bit-for-bit later; it answers
the first question: is the ARCHITECTURE (INT8 bit-serial CIM + LUT split
softmax) accurate enough on attention-shaped data?

Run:  uv run pytest tests -s
"""
import numpy as np
import pytest

from cimple_rtl.model import cimple_model as m

N, DM, D = 16, 64, 32          # tokens, model dim (= CIM rows), head dim


def make_head(seed, w_std=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, (N, DM))
    Wq, Wk, Wv = (rng.normal(0, w_std, (DM, D)) for _ in range(3))
    return X, Wq, Wk, Wv


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run(seed):
    X, Wq, Wk, Wv = make_head(seed)
    ref, _ = m.attention_head_float(X, Wq, Wk, Wv)
    r = m.attention_head_cim(X, Wq, Wk, Wv)
    got = r["out"] * r["out_scale"]
    return ref, got, r


def test_projection_matches_float():
    """Stage 1 alone: INT8 CIM projection vs float X @ W."""
    X, Wq, Wk, Wv = make_head(0)
    _, aux = m.attention_head_float(X, Wq, Wk, Wv)
    r = m.attention_head_cim(X, Wq, Wk, Wv)
    for name in ("Q", "K", "V"):
        scale = np.abs(aux[name]).max() / 127
        err = np.abs(r[name] * scale - aux[name]).max() / np.abs(aux[name]).max()
        print(f"{name} projection max rel err {err:.4f}")
        assert err < 0.05


def test_output_close_to_float_attention():
    """End to end: the head output tracks float attention."""
    cos = []
    for seed in range(5):
        ref, got, _ = run(seed)
        cos.append(cosine(ref, got))
        print(f"seed {seed}: cosine {cos[-1]:.4f}  "
              f"rel L2 {np.linalg.norm(ref - got) / np.linalg.norm(ref):.4f}")
    assert min(cos) > 0.9, f"worst cosine {min(cos):.3f}"


@pytest.mark.xfail(strict=True, reason="open design issue: UQ0.8 exp values vs signed INT8 activations")
def test_exp_values_fit_signed_activation():
    """The e_j values go back into the CIM as ACTIVATIONS for A'V.

    The core treats an activation as signed INT8 (bit 7 is subtracted), but
    the exp LUT produces UQ0.8, i.e. 0..255. Any e >= 128 would be read by the
    RTL as a negative number.

    The model uses e unsigned, so this is NOT visible in the other tests --
    it only shows up once the RTL feeds e into the core.
    """
    over = 0
    total = 0
    for seed in range(5):
        _, _, r = run(seed)
        over += int((r["e"] > 127).sum())
        total += r["e"].size
    print(f"{over}/{total} exp values exceed 127")
    assert over == 0, "exp values >127 would be misread as negative by the core"


def test_softmax_underflow_rate():
    """Rows where the constant-max exp LUT underflows to all zeros."""
    dead = 0
    rows = 0
    for seed in range(5):
        _, _, r = run(seed)
        dead += int((r["e"].sum(axis=1) == 0).sum())
        rows += r["e"].shape[0]
    print(f"{dead}/{rows} rows underflow completely")
    assert dead == 0
