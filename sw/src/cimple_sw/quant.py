"""INT8 quantisation of a Hugging Face causal LM, for the Sec. V-B accuracy study.

The paper evaluates CIMple's softmax on "a pre-trained transformer model
quantized to int8, TinyLlama" and compares against "the same" INT8 model with
PyTorch's softmax (Ahn et al., arXiv:2604.15944v2, Sec. V-B, Fig. 11). It does
not say *how* the model was quantised, so the scheme below is an assumption
chosen to match what the CIM core computes (INT8 weights x INT8 activations):

  "w8a8"  [A] default. Every ``nn.Linear`` inside the decoder layers (q/k/v/o,
          gate/up/down) gets
            * INT8 symmetric per-output-channel weights,
              w_q = clip(round(w / s_w), -128, 127),  s_w = max|w_row| / 127,
            * per-token dynamic symmetric INT8 activations,
              x_q = clip(round(x / s_x), -128, 127),  s_x = max|x_token| / 127.
          Reason: the most common INT8 PTQ recipe that needs no calibration
          data and keeps the 1.1B model usable; per-token activations mirror
          the per-vector quantiser after each CIM pass (Fig. 7).
  "w8"    weight-only INT8 (activations stay fp32).
  "dynamic" torch.ao ``quantize_dynamic`` (real int8 kernels, per-tensor
          activations from fbgemm/onednn); offered for comparison only.
  "none"  no quantisation.

lm_head and the embeddings stay floating point unless ``quantize_lm_head``
[A] -- the paper maps attention on CIMple and is silent on the classifier.

Rounding is ``torch.round`` (half-to-even), the same as the golden model's
``to_int8`` (``np.rint``); scales are clamped away from zero so an all-zero
row/token quantises to zeros instead of NaN.

Fake-quant caveat: the forward pass computes (x_q*s_x) @ (w_q*s_w)^T in fp32.
That equals s_x*s_w*(x_q @ w_q^T) up to fp32 rounding, but a true int32
accumulator is exact: with K up to 5632 the integer dot product can exceed
2**24 and fp32 then rounds it. The error is ~1 ulp of the fp32 result, far
below the INT8 quantisation error itself, so it is not modelled.

Weights are quantised once at load: the INT8 codes and scales are kept as
buffers (``weight_int8``, ``weight_scale``) and the dequantised fp32 weight
is materialised once, so each forward only quantises the activation.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

QuantMode = Literal["w8a8", "w8", "dynamic", "none"]
QUANT_MODES: tuple[str, ...] = ("w8a8", "w8", "dynamic", "none")

QMIN, QMAX = -128, 127
_EPS = 1e-12            # [A] scale floor for all-zero rows / tokens


def quantize_symmetric(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 along ``dim``: returns (int8 codes, fp32 scale, keepdim).

    x ~= codes * scale; scale = max|x| / 127 over ``dim``.
    """
    x = x.float()
    scale = (x.abs().amax(dim=dim, keepdim=True) / QMAX).clamp_min(_EPS)
    q = torch.round(x / scale).clamp_(QMIN, QMAX)
    return q.to(torch.int8), scale


def fake_quant_per_token(x: torch.Tensor) -> torch.Tensor:
    """Per-token (last dim) dynamic symmetric INT8 fake-quant, fp32 out."""
    xf = x.float()
    scale = (xf.abs().amax(dim=-1, keepdim=True) / QMAX).clamp_min(_EPS)
    # |x/scale| <= 127 by construction, so the clip is a no-op kept for safety
    return xf.div(scale).round_().clamp_(QMIN, QMAX).mul_(scale)


class QuantLinear(nn.Module):
    """``nn.Linear`` with pre-quantised INT8 per-channel weights.

    act_quant=True gives W8A8 (per-token dynamic INT8 activations); False
    gives weight-only W8.
    """

    def __init__(self, linear: nn.Linear, act_quant: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant
        w = linear.weight.detach()
        q, s = quantize_symmetric(w, dim=1)                 # per output channel
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", s)             # (out, 1) fp32
        # dequantised once; exact product of an int8 code and an fp32 scale
        self.register_buffer("weight", (q.float() * s).to(w.dtype))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        if self.act_quant:
            x = fake_quant_per_token(x)
        y = nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)
        return y.to(dt)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"w=int8/ch, a={'int8/token' if self.act_quant else 'fp'}")


def _decoder_layers(model: nn.Module) -> nn.Module:
    """The stack of decoder layers (Llama-style ``model.model.layers``)."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("cannot find decoder layers (expected model.model.layers)")
    return layers


def _replace_linears(root: nn.Module, act_quant: bool) -> int:
    n = 0
    for name, child in list(root.named_children()):
        if isinstance(child, nn.Linear):
            setattr(root, name, QuantLinear(child, act_quant=act_quant))
            n += 1
        else:
            n += _replace_linears(child, act_quant)
    return n


def quantize_model(model: nn.Module, mode: QuantMode = "w8a8",
                   quantize_lm_head: bool = False) -> nn.Module:
    """Quantise ``model`` in place (returns it). Idempotent per module.

    Records the mode in ``model.cimple_quant`` and the number of replaced
    linears in ``model.cimple_quant_n``.
    """
    if mode not in QUANT_MODES:
        raise ValueError(f"unknown quant mode {mode!r}; choose from {QUANT_MODES}")
    if getattr(model, "cimple_quant", "none") != "none":
        raise RuntimeError(f"model already quantised ({model.cimple_quant})")
    n = 0
    if mode in ("w8a8", "w8"):
        act = mode == "w8a8"
        n = _replace_linears(_decoder_layers(model), act)
        if quantize_lm_head and isinstance(getattr(model, "lm_head", None), nn.Linear):
            model.lm_head = QuantLinear(model.lm_head, act_quant=act)
            n += 1
    elif mode == "dynamic":
        from torch.ao.quantization import quantize_dynamic
        layers = _decoder_layers(model)
        for i in range(len(layers)):
            layers[i] = quantize_dynamic(layers[i], {nn.Linear}, dtype=torch.qint8)
        n = sum(1 for m in layers.modules() if type(m).__name__ == "Linear"
                and "quantized" in type(m).__module__)
        if quantize_lm_head:
            model.lm_head = quantize_dynamic(nn.Sequential(model.lm_head), {nn.Linear},
                                             dtype=torch.qint8)[0]
            n += 1
    model.cimple_quant = mode
    model.cimple_quant_n = n
    return model


def quantized_linears(model: nn.Module) -> list[tuple[str, QuantLinear]]:
    """All ``QuantLinear`` modules, for inspection / tests."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear)]
