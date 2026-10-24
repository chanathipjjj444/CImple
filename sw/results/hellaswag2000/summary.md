# CIMple softmax accuracy study (Sec. V-B, Fig. 11)

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, quant `w8a8`, 0-shot, limit `2000`, quantize_lm_head `False`, batch `8`, calibrate `False`
- values in %, +- = lm-eval stderr; delta = preset - float (points)
- `float` keeps Q/K/V and the attention output in fp32, so `delta <preset>` includes the per-(batch, head) INT8 attention quantisation, not only the softmax (run `float_int8attn` for a softmax-only delta)
- dynamic per-sequence INT8 scales make the INT8-attention presets' continuation log-probs depend on later tokens (not strictly causal)
- paper column: [A] the paper's metric is assumed to be acc_norm for ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)

| task | metric | float | cimple | delta cimple | paper CIMple | paper delta |
|---|---|---|---|---|---|---|
| HellaSwag | acc | 44.55 +- 1.11 | 44.90 +- 1.11 | +0.35 |  |  |
| HellaSwag | acc_norm | 55.85 +- 1.11 | 55.60 +- 1.11 | -0.25 | 60.52* | -0.54* |

| preset | docs | eval s | docs/s | threads | batch | softmax config |
|---|---|---|---|---|---|---|
| float | 2000 | 2691.6 | 0.743 | 11 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=False, quantize_out=False)` |
| cimple | 2000 | 2845.0 | 0.703 | 11 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |

| preset | task | docs | requests | eval s | requests/s | s/doc |
|---|---|---|---|---|---|---|
| float | HellaSwag | 2000 | 8000 | 2691.6 | 2.97 | 1.346 |
| cimple | HellaSwag | 2000 | 8000 | 2845.0 | 2.81 | 1.422 |
