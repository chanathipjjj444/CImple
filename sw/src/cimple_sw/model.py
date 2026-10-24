"""Load the INT8 TinyLlama used in the paper's accuracy study (Sec. V-B, Fig. 11).

Ahn et al. (arXiv:2604.15944v2) evaluate "a pre-trained transformer model
quantized to int8, TinyLlama" twice: once with PyTorch's softmax (baseline)
and once with CIMple's LUT-based split softmax. :func:`load_model` builds
either from the same checkpoint and the same quantisation
(:mod:`cimple_sw.quant`):

  softmax="float"  -> stock attention (``attn_implementation="eager"`` by
                      default, i.e. ``torch.softmax`` in fp32) on the INT8
                      model; Q/K/V and the attention output stay fp32.
  any other preset -> attention routed through :mod:`cimple_sw.hf_attention`
                      with that :class:`~cimple_sw.config.SoftmaxConfig`
                      (including "float_int8attn": fp32 softmax on INT8
                      Q/K/V with an INT8 output).

"cimple" - "float" therefore measures the LUT softmax PLUS the per-(b,h) INT8
quantisation of Q/K/V and of the attention output; "cimple" -
"float_int8attn" isolates the softmax alone.

[A] Checkpoint: the paper names TinyLlama [26] but not the variant; we
default to ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` (the release whose
published HellaSwag acc_norm, ~60.5, matches the paper's 60.52). The base
``TinyLlama-1.1B-intermediate-step-1431k-3T`` works too via ``name``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .config import PRESETS, SoftmaxConfig

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def resolve_softmax(softmax: SoftmaxConfig | str) -> SoftmaxConfig:
    if isinstance(softmax, SoftmaxConfig):
        return softmax
    if softmax not in PRESETS:
        raise ValueError(f"unknown softmax preset {softmax!r}; choose from {list(PRESETS)}")
    return PRESETS[softmax]


def needs_hook(cfg: SoftmaxConfig) -> bool:
    """True unless ``cfg`` is plain fp32 attention that stock eager/SDPA can run."""
    return cfg.mode != "float" or cfg.quantize_qkv or cfg.quantize_out


def prepare_model(model, softmax: SoftmaxConfig | str = "cimple", quant: str = "w8a8", *,
                  quantize_lm_head: bool = False, tokenizer=None,
                  calibrate: bool = False, calib_texts=None, per_head: bool = False):
    """Quantise an already-loaded HF model and attach the softmax config.

    The model must have been created with ``attn_implementation="cimple"``
    unless ``needs_hook(cfg)`` is False (or it is switched afterwards via
    ``model.set_attn_implementation``). Static score-scale mode always
    calibrates (it cannot run otherwise). ``calibrate=True`` switches a
    split_lut config to ``score_scale="static"`` and calibrates it (so the
    installed scales are actually used); it is a no-op for ``mode="float"``.
    """
    from . import hf_attention as hfa
    from .quant import quantize_model

    cfg = resolve_softmax(softmax)
    if calibrate and cfg.mode != "float":
        cfg = cfg.with_(score_scale="static")
    quantize_model(model, quant, quantize_lm_head=quantize_lm_head)
    hfa.set_softmax_config(model, cfg)
    model.eval()
    if cfg.mode != "float" and cfg.score_scale == "static":
        if tokenizer is None:
            raise ValueError("calibration needs a tokenizer")
        hfa.calibrate_static_scales(model, tokenizer, calib_texts, per_head=per_head)
    return model


def load_model(name: str = DEFAULT_MODEL, quant: str = "w8a8",
               softmax: SoftmaxConfig | str = "cimple", dtype: torch.dtype = torch.float32,
               calibrate: bool = False, *, baseline_attn: str = "eager",
               quantize_lm_head: bool = False, per_head: bool = False,
               calib_texts=None, local_files_only: Optional[bool] = None):
    """Load ``name``, quantise it, and route attention per ``softmax``.

    Returns (model, tokenizer). ``baseline_attn`` picks the stock kernel used
    for the float preset ("eager" = explicit torch.softmax; "sdpa" is faster
    and numerically equivalent to ~1e-6).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import hf_attention as hfa

    cfg = resolve_softmax(softmax)
    attn = hfa.register() if needs_hook(cfg) else baseline_attn
    kw = {} if local_files_only is None else {"local_files_only": local_files_only}
    tok = AutoTokenizer.from_pretrained(name, **kw)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn,
                                                 low_cpu_mem_usage=True, **kw)
    prepare_model(model, cfg, quant, quantize_lm_head=quantize_lm_head, tokenizer=tok,
                  calibrate=calibrate, calib_texts=calib_texts, per_head=per_head)
    return model, tok
