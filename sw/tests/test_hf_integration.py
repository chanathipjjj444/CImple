"""HF / PyTorch integration of the CIMple softmax (Sec. V-B accuracy study).

Fast tests use a tiny randomly initialised Llama (no download). The TinyLlama
smoke test is marked slow.
"""
from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from cimple_sw import hf_attention as hfa
from cimple_sw.config import PRESETS, SoftmaxConfig
from cimple_sw.model import prepare_model
from cimple_sw.quant import QuantLinear, quantize_model, quantized_linears

HAVE_SPLIT = importlib.util.find_spec("cimple_sw.split_softmax") is not None
needs_split = pytest.mark.skipif(not HAVE_SPLIT, reason="cimple_sw.split_softmax not present")


def tiny_llama(attn: str, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    if attn == hfa.ATTN_NAME:
        hfa.register()
    torch.manual_seed(seed)
    m = LlamaForCausalLM._from_config(cfg, attn_implementation=attn).eval()
    # random init is too tame to stress the softmax; widen the q/k projections
    with torch.no_grad():
        for layer in m.model.layers:
            layer.self_attn.q_proj.weight.mul_(8)
            layer.self_attn.k_proj.weight.mul_(8)
    return m


def pair(softmax, quant="none"):
    """(eager reference, custom-attention model) with identical weights."""
    ref = tiny_llama("eager")
    cus = tiny_llama(hfa.ATTN_NAME)
    cus.load_state_dict(ref.state_dict())
    quantize_model(ref, quant)
    prepare_model(cus, softmax, quant)
    return ref, cus


IDS = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(1))


@torch.no_grad()
def test_float_mode_matches_eager():
    ref, cus = pair(PRESETS["float"])
    a, b = ref(IDS).logits, cus(IDS).logits
    assert (a - b).abs().max().item() < 1e-5


@torch.no_grad()
def test_float_mode_matches_eager_on_quantised_model():
    ref, cus = pair("float", quant="w8a8")
    assert (ref(IDS).logits - cus(IDS).logits).abs().max().item() < 1e-5


@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_padding_and_causal(side):
    """Padded batch == each sequence run alone (on its real tokens)."""
    _, cus = pair(PRESETS["float"])
    ref, _ = pair(PRESETS["float"])
    lens = [24, 15]
    ids = IDS.clone()
    am = torch.zeros_like(ids)
    for i, n in enumerate(lens):
        if side == "left":
            am[i, -n:] = 1
        else:
            am[i, :n] = 1
    pos = (am.cumsum(-1) - 1).clamp_min(0)
    out = cus(ids, attention_mask=am, position_ids=pos).logits
    ref_out = ref(ids, attention_mask=am, position_ids=pos).logits
    for i, n in enumerate(lens):
        sl = slice(-n, None) if side == "left" else slice(0, n)
        alone = cus(ids[i:i + 1, sl]).logits[0]
        assert (out[i, sl] - alone).abs().max().item() < 1e-4
        assert (out[i, sl] - ref_out[i, sl]).abs().max().item() < 1e-5
    assert torch.isfinite(out).all()


def test_keep_mask_formats():
    b, lq, lk = 2, 3, 5
    causal = hfa.keep_mask(None, b, lq, lk)
    expect = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(causal[0, 0], expect)
    add = torch.zeros(b, 1, lq, lk).masked_fill(~expect, torch.finfo(torch.float32).min)
    assert torch.equal(hfa.keep_mask(add, b, lq, lk)[0, 0], expect)
    inf = torch.zeros(b, 1, lq, lk).masked_fill(~expect, float("-inf"))
    assert torch.equal(hfa.keep_mask(inf, b, lq, lk)[0, 0], expect)
    assert torch.equal(hfa.keep_mask(expect.view(1, 1, lq, lk), b, lq, lk)[0, 0], expect)
    with pytest.raises(ValueError):
        hfa.keep_mask(add + 0.5, b, lq, lk)


def test_quantised_linears_are_int8():
    m = tiny_llama("eager")
    quantize_model(m, "w8a8")
    ql = quantized_linears(m)
    assert len(ql) == 2 * 7                       # q,k,v,o,gate,up,down per layer
    assert not isinstance(m.lm_head, QuantLinear)
    for _, q in ql:
        assert q.weight_int8.dtype == torch.int8
        # the fp32 weight used in forward IS code * scale, bit for bit
        assert torch.equal(q.weight, q.weight_int8.float() * q.weight_scale)
        assert torch.equal((q.weight / q.weight_scale).round(), q.weight_int8.float())
        assert q.weight_int8.abs().amax(dim=1).eq(127).all()  # per-channel absmax -> 127
    with pytest.raises(RuntimeError):
        quantize_model(m, "w8a8")                 # applied once only


def test_w8a8_activation_is_per_token_int8():
    from cimple_sw.quant import fake_quant_per_token
    x = torch.randn(3, 5, 64)
    y = fake_quant_per_token(x)
    s = x.abs().amax(-1, keepdim=True) / 127
    codes = y / s
    assert (codes - codes.round()).abs().max() < 1e-4
    assert codes.abs().max() <= 127 + 1e-4


@needs_split
@pytest.mark.parametrize("preset,tv_max", [("cimple", 0.03), ("cimple_hw8", 0.3)])
@torch.no_grad()
def test_cimple_preset_close_to_float(preset, tv_max):
    """The approximation changes logits a little. hw8 is looser: with the
    constant z_quant_max, 8b exp entries of rows whose max score sits far
    below 127*s underflow to 0 (the paper's design, stressed here by the x8
    q/k weights of the toy model)."""
    ref, cus = pair(preset, quant="w8a8")
    a, b = ref(IDS).logits, cus(IDS).logits
    diff = (a - b).abs().max().item()
    p_ref, p_cus = a.softmax(-1), b.softmax(-1)
    tv = 0.5 * (p_ref - p_cus).abs().sum(-1).max().item()
    print(f"\n[{preset}] max |dlogit| = {diff:.4g}, max TV(next-token) = {tv:.4g}, "
          f"logit range = {a.abs().max().item():.3g}")
    assert diff > 0                               # the approximation is really active
    assert tv < tv_max
    assert torch.isfinite(b).all()


@needs_split
@torch.no_grad()
def test_static_calibration_populates_every_layer():
    cfg = PRESETS["cimple"].with_(score_scale="static")
    cus = tiny_llama(hfa.ATTN_NAME)

    class Tok:                                   # byte-level stand-in tokenizer
        def __call__(self, text, return_tensors=None, truncation=True, max_length=64):
            ids = torch.tensor([list(text.encode()[:max_length])])
            return {"input_ids": ids}

    prepare_model(cus, cfg, "w8a8", tokenizer=Tok())
    mods = hfa.attention_modules(cus)
    assert len(mods) == 2
    for m in mods:
        s = m.cimple_score_scale
        assert s.shape == (1, 1, 1, 1) and torch.isfinite(s).all() and (s > 0).all()
        assert hfa.STATIC_SCALES[m.layer_idx] is s
    assert torch.isfinite(cus(IDS).logits).all()
    per_head = hfa.calibrate_static_scales(cus, Tok(), per_head=True)
    assert all(v.shape == (1, 4, 1, 1) for v in per_head.values())


@torch.no_grad()
def test_calibration_observer_float_mode():
    """Observer path works without split_softmax (local absmax fallback)."""
    _, cus = pair(PRESETS["float"])
    with hfa.observing() as obs:
        cus(IDS)
    assert sorted(obs) == [0, 1] and all(v.shape == (4,) for v in obs.values())


# ---------------------------------------------------------------------------
@pytest.mark.slow
@needs_split
@torch.no_grad()
def test_tinyllama_smoke():
    from cimple_sw.model import load_model
    text = ("The capital of France is Paris. It is known for the Eiffel Tower, "
            "museums such as the Louvre, and its cafes along the river Seine.")
    losses = {}
    for preset in ("float", "cimple"):
        model, tok = load_model(softmax=preset)
        enc = tok(text, return_tensors="pt")
        losses[preset] = model(**enc, labels=enc["input_ids"]).loss.item()
        del model
    print(f"\nTinyLlama w8a8 loss: {losses}  ppl: "
          f"{ {k: round(math.exp(v), 3) for k, v in losses.items()} }")
    assert losses["float"] < 4.0
    assert abs(losses["cimple"] - losses["float"]) < 0.25


@needs_split
@torch.no_grad()
def test_right_padded_batch_is_batch_invariant_with_sequence_lengths():
    """lm-eval style: right padding with token 0, no attention mask."""
    _, cus = pair("cimple", quant="w8a8")
    ids = IDS.clone()
    n = 13
    ids[1, n:] = 0
    alone = cus(ids[1:2, :n]).logits[0]
    with hfa.sequence_lengths([24, n]):
        out = cus(ids).logits
    assert torch.equal(out[1, :n], alone)            # bit-identical to running alone
    assert torch.equal(out[0], cus(ids[0:1]).logits[0])
    leaky = cus(ids).logits[1, :n]                   # without it, pads enter the scales
    print(f"\npad leakage without sequence_lengths: {(leaky - alone).abs().max().item():.3g}")


def test_merge_summary(tmp_path):
    import json

    from cimple_sw import eval as ev

    def fake(preset, arc, wg):
        return {"preset": preset, "softmax_config": preset, "model": "m", "quant": "w8a8",
                "num_fewshot": 0, "limit": 4, "batch_size": 1, "threads": 1, "load_s": 0,
                "eval_s": 1.0, "n_docs": 8, "docs_per_s": 8.0,
                "results": {"arc_challenge": {"acc,none": arc, "acc_stderr,none": 0.01,
                                              "acc_norm,none": arc, "acc_norm_stderr,none": 0.01},
                            "winogrande": {"acc,none": wg, "acc_stderr,none": 0.02}}}
    (tmp_path / "float.json").write_text(json.dumps(fake("float", 0.30, 0.50)))
    (tmp_path / "cimple.json").write_text(json.dumps(fake("cimple", 0.31, 0.49)))
    md = ev.merge(tmp_path)
    assert (tmp_path / "summary.md").read_text() == md
    assert "| ARC-Challenge | acc_norm | 30.00 +- 1.00 | 31.00 +- 1.00 | +1.00 | 31.74* | +0.60* |" in md
    assert "| Winogrande | acc | 50.00 +- 2.00 | 49.00 +- 2.00 | -1.00 | 57.22* | -0.32* |" in md


def test_cli_luts_matches_golden_model(capsys):
    from cimple_rtl.model import cimple_model as gm

    from cimple_sw.cli import main
    assert main(["luts", "--score-scale", "0.05"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    assert [int(x, 16) for x in lines[:256]] == gm.build_exp_lut(0.05).tolist()
    assert [int(x, 16) for x in lines[256:]] == gm.build_recip_lut().tolist()
    assert main(["bogus"]) == 2


# ---------------------------------------------------------------------------
# regression tests for the hf_eval review findings
# ---------------------------------------------------------------------------
def _stub(cfg, layer_idx=0, scaling=16 ** -0.5):
    m = torch.nn.Module()
    m.cimple_cfg, m.layer_idx, m.scaling = cfg, layer_idx, scaling
    return m


def _rand_qkv(b=1, h=2, l=6, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(3 * torch.randn(b, h, l, d, generator=g) for _ in range(3))


@needs_split
@torch.no_grad()
def test_hook_float_mode_honours_quantize_qkv_like_kernel():
    """F1: mode='float' + quantize_qkv runs the fp32 softmax on INT8 Q/K/V
    through the hook, exactly as split_softmax.cimple_attention does."""
    from cimple_sw.split_softmax import cimple_attention
    q, k, v = _rand_qkv()
    keep = hfa.keep_mask(None, 1, 6, 6)
    plain = PRESETS["float"]
    fq = plain.with_(quantize_qkv=True)
    out_plain, _ = hfa.cimple_attention_forward(_stub(plain), q, k, v, None)
    out_fq, probs = hfa.cimple_attention_forward(_stub(fq), q, k, v, None)
    ref = cimple_attention(q, k, v, scaling=16 ** -0.5, keep=keep, cfg=fq)
    assert not torch.equal(out_plain, out_fq)                 # the flag is not ignored
    assert torch.allclose(out_fq, ref.transpose(1, 2), atol=1e-6, rtol=0)
    assert probs is not None and torch.allclose(probs.sum(-1), torch.ones(1, 2, 6))
    # quantize_out: the output is INT8 in V's per-(b,h) scale
    from cimple_sw.split_softmax import _quant_per_head
    out_q, _ = hfa.cimple_attention_forward(_stub(PRESETS["float_int8attn"]), q, k, v, None)
    _, sv = _quant_per_head(v)
    codes = out_q.transpose(1, 2).double() / sv
    assert (codes - codes.round()).abs().max() < 1e-4 and codes.abs().max() <= 128
    assert not torch.equal(out_q, out_fq)


@needs_split
@torch.no_grad()
def test_float_int8attn_preset_through_model_and_padding():
    """F1/F2: the matched baseline differs from plain float on a W8A8 model,
    is routed through the hook by load_model, and keeps pads out of its scales."""
    from cimple_sw.model import needs_hook
    assert not needs_hook(PRESETS["float"])
    assert needs_hook(PRESETS["float_int8attn"]) and needs_hook(PRESETS["cimple"])
    ref, cus = pair("float_int8attn", quant="w8a8")
    a, b = ref(IDS).logits, cus(IDS).logits
    assert (a - b).abs().max().item() > 0
    assert torch.isfinite(b).all()
    ids = IDS.clone()
    n = 13
    ids[1, n:] = 0
    alone = cus(ids[1:2, :n]).logits[0]
    with hfa.sequence_lengths([24, n]):
        out = cus(ids).logits
    assert torch.allclose(out[1, :n], alone, atol=1e-5, rtol=0)


class _ByteTok:                                   # byte-level stand-in tokenizer
    def __call__(self, text, return_tensors=None, truncation=True, max_length=64):
        return {"input_ids": torch.tensor([list(text.encode()[:max_length])])}


@needs_split
@torch.no_grad()
def test_calibrate_flag_switches_to_static_scales():
    """F3: --calibrate (prepare_model(calibrate=True)) must use the scales."""
    dyn = tiny_llama(hfa.ATTN_NAME)
    cal = tiny_llama(hfa.ATTN_NAME)
    prepare_model(dyn, "cimple", "w8a8")
    prepare_model(cal, "cimple", "w8a8", tokenizer=_ByteTok(), calibrate=True)
    assert dyn.cimple_cfg.score_scale == "dynamic"
    assert cal.cimple_cfg.score_scale == "static"
    assert all(m.cimple_cfg.score_scale == "static" for m in hfa.attention_modules(cal))
    assert not torch.equal(dyn(IDS).logits, cal(IDS).logits)
    flt = tiny_llama(hfa.ATTN_NAME)
    prepare_model(flt, "float", "w8a8", tokenizer=_ByteTok(), calibrate=True)
    assert flt.cimple_cfg == PRESETS["float"]                # no-op for float mode


@needs_split
@torch.no_grad()
def test_dynamic_scales_make_logprobs_depend_on_later_tokens():
    """F5 (documented limitation): with dynamic per-sequence INT8 scales the
    prediction at position t changes with the token at t+1; float and the
    static-score / fp32-QKV configuration stay strictly causal."""
    ids = torch.randint(0, 256, (1, 21), generator=torch.Generator().manual_seed(3))
    alt = ids.clone()
    alt[0, 20] = (ids[0, 20] + 97) % 256
    causal_cfg = PRESETS["cimple"].with_(score_scale="static", quantize_qkv=False)
    deltas = {}
    for name, cfg in (("float", PRESETS["float"]), ("cimple", PRESETS["cimple"]),
                      ("causal", causal_cfg)):
        m = tiny_llama(hfa.ATTN_NAME)
        prepare_model(m, cfg, "w8a8", tokenizer=_ByteTok())
        la = m(ids).logits[0, 19].log_softmax(-1)
        lb = m(alt).logits[0, 19].log_softmax(-1)
        deltas[name] = (la - lb).abs().max().item()
    assert deltas["float"] == 0.0 and deltas["causal"] == 0.0
    assert deltas["cimple"] > 0.0
    from pathlib import Path
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "The attention weights never do" not in readme
    assert "ctx + cont[:-1]" in readme


def _fake_run(preset, arc, **over):
    d = {"preset": preset, "softmax_config": preset, "model": "m", "quant": "w8a8",
         "quantize_lm_head": False, "calibrate": False,
         "num_fewshot": 0, "limit": 4, "batch_size": 1, "threads": 1, "load_s": 0,
         "eval_s": 1.0, "n_docs": 8, "docs_per_s": 8.0,
         "results": {"arc_challenge": {"acc_norm,none": arc, "acc_norm_stderr,none": 0.01}}}
    d.update(over)
    return d


@pytest.mark.parametrize("field,a,b", [("quantize_lm_head", False, True),
                                       ("calibrate", False, True),
                                       ("batch_size", 8, 16)])
def test_merge_warns_on_mismatched_settings(tmp_path, field, a, b):
    """F4/F6: lm_head quantisation, calibration and batch size must match."""
    import json

    from cimple_sw import eval as ev
    (tmp_path / "float.json").write_text(json.dumps(_fake_run("float", 0.30, **{field: a})))
    (tmp_path / "cimple.json").write_text(json.dumps(_fake_run("cimple", 0.31, **{field: b})))
    assert "WARNING" in ev.merge(tmp_path)
    (tmp_path / "cimple.json").write_text(json.dumps(_fake_run("cimple", 0.31, **{field: a})))
    assert "WARNING" not in ev.merge(tmp_path)


def test_merge_old_json_without_calibrate_field(tmp_path):
    import json

    from cimple_sw import eval as ev
    old = _fake_run("float", 0.30)
    del old["calibrate"]
    (tmp_path / "float.json").write_text(json.dumps(old))
    (tmp_path / "cimple.json").write_text(json.dumps(_fake_run("cimple", 0.31)))
    assert "WARNING" not in ev.merge(tmp_path)


def test_merge_reports_delta_vs_matched_baseline(tmp_path):
    """F2: with float_int8attn present, the softmax-only delta is reported."""
    import json

    from cimple_sw import eval as ev
    for p, v in (("float", 0.30), ("float_int8attn", 0.305), ("cimple", 0.31)):
        (tmp_path / f"{p}.json").write_text(json.dumps(_fake_run(p, v)))
    md = ev.merge(tmp_path)
    hdr = next(ln for ln in md.splitlines() if ln.startswith("| task |"))
    assert hdr.split(" | ")[2:5] == ["float", "float_int8attn", "cimple"]
    assert "vs float_int8attn cimple" in hdr
    assert ("| ARC-Challenge | acc_norm | 30.00 +- 1.00 | 30.50 +- 1.00 | 31.00 +- 1.00 "
            "| +0.50 | +1.00 | +0.50 | 31.74* | +0.60* |") in md
    assert "includes the per-(batch, head) INT8 attention quantisation" in md
