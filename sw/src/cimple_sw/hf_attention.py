"""Hugging Face ``transformers`` hook that runs attention through CIMple's softmax.

The paper evaluates its LUT-based split softmax by "writing [it] quantized to
int8 in PyTorch and evaluating it with a pre-trained transformer model
quantized to int8, TinyLlama" (Ahn et al., arXiv:2604.15944v2, Sec. V-B).
This module is the glue: it registers an attention implementation named
``"cimple"`` with ``transformers.AttentionInterface`` so every
``LlamaAttention`` layer calls :func:`cimple_attention_forward` instead of
eager/SDPA, and the numerics live in ``cimple_sw.split_softmax``.

Mask format (transformers 4.57, ``masking_utils.create_causal_mask``): the mask
builder is looked up by the same name in ``AttentionMaskInterface``. We
register the SDPA builder with ``allow_is_causal_skip=False``, so the layer
receives a bool ``(B, 1, Lq, Lk)`` mask, True = attend. The forward still
accepts every other format robustly: an additive float mask (0 = attend,
large negative = masked), a bool mask, or ``None`` (pure causal, aligned to
the bottom-right so cached decoding works).

Rows with no visible key (left-padding query positions) output zeros (the
float path uses a dummy row internally so no NaN escapes; the CIMple kernel
returns 0 for an empty row by itself). Right-padded batches without a mask
(lm-evaluation-harness) are handled by :func:`sequence_lengths`.

``mode="float"`` honours ``quantize_qkv`` / ``quantize_out`` like the kernel:
with ``quantize_qkv`` the fp32 softmax runs on per-(b,h) INT8 Q/K/V and, with
``quantize_out``, the output is requantised to INT8 in V's scale -- the
``float_int8attn`` preset, i.e. the cimple preset with only the softmax
swapped for fp32. Without ``quantize_qkv`` it is plain fp32 attention.

Per-layer configuration: each attention module carries ``cimple_cfg``
(a :class:`~cimple_sw.config.SoftmaxConfig`) and, for
``score_scale == "static"``, ``cimple_score_scale`` (tensor broadcastable to
(B, H, 1, 1)). :data:`STATIC_SCALES` mirrors the latter as a global registry
``layer_idx -> scale`` for the most recently calibrated model. Calibration
(:func:`calibrate_static_scales`) runs a few forwards over a FIXED built-in
text -- never the evaluation data -- and stores max|score| / z_quant_max per
layer ([A] per layer by default: one scale register / one exp LUT per layer,
matching the hardware's single quantiser scale; per-head is an option).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch
from torch import nn

from .config import PRESETS, SoftmaxConfig

ATTN_NAME = "cimple"

# layer_idx -> static score scale (tensor broadcastable to (B,H,1,1)) [global registry]
STATIC_SCALES: dict[int, torch.Tensor] = {}

# calibration observer: layer_idx -> running absmax (H,) ; None when not observing
_OBSERVER: Optional[dict[int, torch.Tensor]] = None

# per-sequence valid lengths of a right-padded batch (set by sequence_lengths())
_SEQ_LENS: Optional[list[int]] = None

_DEFAULT_CFG = PRESETS["cimple"]

# [A] fixed calibration corpus, written for this repo (generic encyclopaedic
# prose on mixed topics). It shares no text with ARC / HellaSwag / OBQA /
# Winogrande, so calibrating on it cannot leak evaluation data.
CALIBRATION_TEXTS: tuple[str, ...] = (
    "The river begins as a thin stream high in the mountains, fed by melting snow "
    "each spring. As it descends it collects water from dozens of smaller creeks, "
    "and by the time it reaches the plains it is wide enough to carry barges loaded "
    "with grain, timber and coal. Towns grew along its banks because the water "
    "provided transport, irrigation and power for the early mills.",
    "To compute the area of a triangle, multiply the length of its base by its "
    "height and divide the result by two. For example, a triangle with a base of "
    "ten centimetres and a height of six centimetres has an area of thirty square "
    "centimetres. The same idea extends to any polygon, which can be split into "
    "triangles whose areas are then added together.",
    "When Maria opened the old wooden box she found a bundle of letters tied with a "
    "faded ribbon. Most were written by her grandmother during the years she worked "
    "as a nurse in a small coastal hospital. The letters described long night shifts, "
    "storms that cut the power, and the quiet satisfaction of seeing patients walk "
    "home after weeks of care.",
    "Plants convert light into chemical energy through photosynthesis. Inside the "
    "chloroplasts, chlorophyll absorbs mostly red and blue light, which is why leaves "
    "look green. The energy is used to split water molecules, releasing oxygen, and "
    "to build sugars from carbon dioxide. Those sugars feed the plant and, through "
    "food chains, nearly every animal on land.",
    "A good recipe for bread needs only flour, water, salt and yeast. Mix them, knead "
    "the dough until it is smooth and elastic, then let it rise in a warm place until "
    "it doubles in size. Shape the loaf, let it rise again, and bake it in a hot oven "
    "until the crust is brown and the bottom sounds hollow when tapped.",
    "Computers store numbers in binary, using only the digits zero and one. An eight "
    "bit integer can represent 256 different values, from minus 128 to 127 when it "
    "is signed. Engineers choose the number of bits as a trade-off: more bits give "
    "more precision, but they cost more memory, more wires and more energy for every "
    "operation the chip performs.",
)


# ---------------------------------------------------------------------------
# mask handling
# ---------------------------------------------------------------------------
def keep_mask(attention_mask: Optional[torch.Tensor], b: int, lq: int, lk: int,
              device=None) -> torch.Tensor:
    """Any HF mask -> bool keep mask broadcastable to (B, 1|H, Lq, Lk), True=attend.

    None means pure causal, aligned bottom-right (query i sees keys
    j <= i + Lk - Lq), which is also correct with a KV cache.
    """
    if attention_mask is None:
        i = torch.arange(lq, device=device).unsqueeze(1)
        j = torch.arange(lk, device=device).unsqueeze(0)
        return (j <= i + (lk - lq)).view(1, 1, lq, lk)
    m = attention_mask
    if m.dim() == 2:                                 # (Lq, Lk)
        m = m.view(1, 1, *m.shape)
    elif m.dim() == 3:                               # (B, Lq, Lk)
        m = m.unsqueeze(1)
    m = m[..., :lk]
    if m.dtype == torch.bool:
        return m
    # additive mask: 0 = attend, very negative / -inf = masked
    keep = m > torch.finfo(m.dtype).min / 2 if m.is_floating_point() else m != 0
    if m.is_floating_point() and torch.any(m[keep] != 0):
        raise ValueError("additive attention bias (non 0/-inf mask) is not supported "
                         "by the CIMple softmax")
    return keep


def _register_mask_fn() -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

    def cimple_mask(*args, **kwargs):
        kwargs["allow_is_causal_skip"] = False       # always materialise a bool mask
        return sdpa_mask(*args, **kwargs)

    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_NAME, cimple_mask)


# ---------------------------------------------------------------------------
# the attention function
# ---------------------------------------------------------------------------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, l, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)


def score_absmax(q, k, *, scaling, keep=None) -> torch.Tensor:
    """(B,H) absmax of kept scores; local fallback for observe_score_absmax."""
    s = (q.float() @ k.float().transpose(-1, -2)) * scaling
    if keep is not None:
        s = s.masked_fill(~keep, 0.0)
    return s.abs().amax(dim=(-1, -2))


def _float_attention(q, k, v, scaling, keep):
    s = (q @ k.transpose(-1, -2)) * scaling
    s = s.masked_fill(~keep, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v, p


def _float_qkv_attention(q, k, v, scaling, keep, cfg: SoftmaxConfig):
    """fp32 softmax on INT8 Q/K/V -- the matched baseline for the split_lut presets.

    Q/K/V are fake-quantised per (batch, head) exactly as
    ``split_softmax.cimple_attention`` does for ``mode="float"`` with
    ``quantize_qkv``. With ``cfg.quantize_out`` the output is also requantised
    to INT8 in V's scale (round-half-away + saturation, as the "full"-table
    split_lut path does), so only the softmax differs from the cimple preset.
    """
    from .luts import round_half_away
    from .split_softmax import _quant_per_head
    (qi, sq), (ki, sk), (vi, sv) = map(_quant_per_head, (q, k, v))
    qf, kf, vf = ((c.to(torch.float64) * sc).float() for c, sc in ((qi, sq), (ki, sk), (vi, sv)))
    out, p = _float_attention(qf, kf, vf, scaling, keep)
    if cfg.quantize_out:
        codes = round_half_away(out.to(torch.float64) / sv).clamp(-128, 127)
        out = (codes * sv).to(out.dtype)
    return out, p


def cimple_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                             value: torch.Tensor, attention_mask: Optional[torch.Tensor],
                             scaling: Optional[float] = None, dropout: float = 0.0,
                             **kwargs):
    """``AttentionInterface`` signature. query (B,Hq,Lq,D), key/value (B,Hkv,Lk,D).

    Returns (attn_output (B,Lq,Hq,D), attn_weights or None). Dropout is
    ignored (evaluation only).
    """
    cfg: SoftmaxConfig = getattr(module, "cimple_cfg", None) or _DEFAULT_CFG
    b, hq, lq, d = query.shape
    lk = key.shape[-2]
    n_rep = hq // key.shape[1]
    if scaling is None:
        scaling = getattr(module, "scaling", d ** -0.5)
    dt = query.dtype
    q = query.float()
    k = _repeat_kv(key, n_rep).float()
    v = _repeat_kv(value, n_rep).float()

    keep = keep_mask(attention_mask, b, lq, lk, device=q.device)
    keep = keep.expand(b, keep.shape[1], lq, lk)
    has_any = keep.any(dim=-1, keepdim=True)                     # (B,1|H,Lq,1)
    keep = keep | ~has_any                                       # dummy row for empty rows

    layer_idx = getattr(module, "layer_idx", None)
    if _OBSERVER is not None and layer_idx is not None:
        try:
            from .split_softmax import observe_score_absmax
        except ImportError:
            observe_score_absmax = score_absmax
        am = observe_score_absmax(q, k, scaling=scaling, keep=keep & has_any)
        am = am.detach().float().amax(dim=0)                    # (H,)
        prev = _OBSERVER.get(layer_idx)
        _OBSERVER[layer_idx] = am if prev is None else torch.maximum(prev, am)

    probs = None
    lens = _SEQ_LENS
    per_seq = lens is not None and len(lens) == b and lq == lk
    if cfg.mode == "float" and not cfg.quantize_qkv:
        out, probs = _float_attention(q, k, v, scaling, keep)
    elif cfg.mode == "float":
        # INT8 Q/K/V (+ INT8 out) with an fp32 softmax. Like the kernel, the
        # per-(b,h) scales must not see pad tokens of a right-padded batch.
        if per_seq:
            out = q.new_zeros(b, hq, lq, d)
            probs = q.new_zeros(b, hq, lq, lk)
            kp = keep.expand(b, hq, lq, lk)
            for i, n in enumerate(lens):
                if n == 0:
                    continue
                o, p = _float_qkv_attention(q[i:i + 1, :, :n], k[i:i + 1, :, :n],
                                            v[i:i + 1, :, :n], scaling,
                                            kp[i:i + 1, :, :n, :n], cfg)
                out[i:i + 1, :, :n] = o
                probs[i:i + 1, :, :n, :n] = p
        else:
            out, probs = _float_qkv_attention(q, k, v, scaling, keep, cfg)
    else:
        keep = keep & has_any               # the kernel outputs 0 for empty rows itself
        from .split_softmax import cimple_attention
        score_scale = None
        if cfg.score_scale == "static" and _OBSERVER is None:
            score_scale = getattr(module, "cimple_score_scale", None)
            if score_scale is None and layer_idx is not None:
                score_scale = STATIC_SCALES.get(layer_idx)
            if score_scale is None:
                raise RuntimeError(f"layer {layer_idx}: static score scale not calibrated; "
                                   "run calibrate_static_scales() first")
        elif cfg.score_scale == "static":               # calibrating: behave dynamically
            cfg = cfg.with_(score_scale="dynamic")
        if per_seq:
            # padded batch: run each sequence on its real prefix only, so the
            # per-(b,h) INT8 / score scales never see pad tokens
            out = q.new_zeros(b, hq, lq, d)
            kp = keep.expand(b, hq, lq, lk)
            for i, n in enumerate(lens):
                if n == 0:
                    continue
                ss = score_scale
                if torch.is_tensor(ss) and ss.dim() == 4 and ss.shape[0] == b:
                    ss = ss[i:i + 1]
                out[i:i + 1, :, :n] = cimple_attention(
                    q[i:i + 1, :, :n], k[i:i + 1, :, :n], v[i:i + 1, :, :n], scaling=scaling,
                    keep=kp[i:i + 1, :, :n, :n], cfg=cfg, score_scale=ss)
        else:
            out = cimple_attention(q, k, v, scaling=scaling, keep=keep, cfg=cfg,
                                   score_scale=score_scale)
    out = out.masked_fill(~has_any, 0.0)
    if probs is not None:
        probs = probs.masked_fill(~has_any, 0.0).to(dt)
    return out.transpose(1, 2).contiguous().to(dt), probs


def register() -> str:
    """Register the "cimple" attention + mask functions (idempotent). Returns the name."""
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN_NAME, cimple_attention_forward)
    _register_mask_fn()
    return ATTN_NAME


# ---------------------------------------------------------------------------
# per-model configuration
# ---------------------------------------------------------------------------
def attention_modules(model: nn.Module) -> list[nn.Module]:
    """Modules that call the attention interface (have ``layer_idx`` and ``scaling``)."""
    return [m for m in model.modules()
            if hasattr(m, "layer_idx") and hasattr(m, "scaling") and hasattr(m, "q_proj")]


def set_softmax_config(model: nn.Module, cfg: SoftmaxConfig | str) -> SoftmaxConfig:
    """Attach ``cfg`` to every attention layer of ``model``."""
    if isinstance(cfg, str):
        cfg = PRESETS[cfg]
    for m in attention_modules(model):
        m.cimple_cfg = cfg
    model.cimple_cfg = cfg
    return cfg


def set_static_scales(model: nn.Module, scales: dict[int, torch.Tensor]) -> None:
    """Install per-layer static score scales and mirror them in STATIC_SCALES."""
    STATIC_SCALES.clear()
    for m in attention_modules(model):
        s = scales[m.layer_idx]
        m.cimple_score_scale = s
        STATIC_SCALES[m.layer_idx] = s


@contextmanager
def sequence_lengths(lens: Optional[Iterable[int]]):
    """Declare the real (unpadded) length of each row of a right-padded batch.

    lm-evaluation-harness right-pads loglikelihood batches and calls the model
    without an attention mask. Pad tokens at the end never influence real
    tokens through causal attention, but they WOULD enter CIMple's per-(b,h)
    dynamic INT8 / score scales; inside this context each sequence is
    processed on its own prefix instead, so pad tokens never enter the scales.
    Results are then batch-size invariant up to fp32 BLAS rounding in the
    linear layers (~1e-6, shape dependent), which INT8 rounding in the kernel
    can amplify to ~1e-3 nats per token -- enough to flip a near-tie answer,
    so compare presets at the same batch size.
    """
    global _SEQ_LENS
    prev, _SEQ_LENS = _SEQ_LENS, (None if lens is None else [int(n) for n in lens])
    try:
        yield
    finally:
        _SEQ_LENS = prev


@contextmanager
def observing():
    """Record per-layer, per-head score absmax during the enclosed forwards."""
    global _OBSERVER
    _OBSERVER = {}
    try:
        yield _OBSERVER
    finally:
        _OBSERVER = None


@torch.no_grad()
def calibrate_static_scales(model: nn.Module, tokenizer, texts: Optional[Iterable[str]] = None,
                            *, max_length: int = 512, per_head: bool = False,
                            install: bool = True) -> dict[int, torch.Tensor]:
    """Static score scale per layer (or per layer/head) = absmax / z_quant_max.

    Runs one forward per text with the observer active; the softmax used
    meanwhile is the model's own config with a dynamic scale. Returns
    {layer_idx: tensor of shape (1,1,1,1) or (1,H,1,1)} and, if ``install``,
    attaches it to the model.
    """
    texts = list(CALIBRATION_TEXTS if texts is None else texts)
    cfg = getattr(model, "cimple_cfg", _DEFAULT_CFG)
    zq = cfg.z_quant_max
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with observing() as obs:
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            model(**{k: v.to(dev) for k, v in enc.items()})
    model.train(was_training)
    if not obs:
        raise RuntimeError("no attention layer was observed -- is attn_implementation 'cimple'?")
    scales = {}
    for li, am in obs.items():
        a = am if per_head else am.amax().reshape(1)
        s = (a.clamp_min(1e-8) / zq).view(1, -1, 1, 1)
        scales[li] = s
    if install:
        set_static_scales(model, scales)
    return scales
