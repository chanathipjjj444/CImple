# CIMple softmax accuracy study (Sec. V-B, Fig. 11)

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, quant `w8a8`, 0-shot, limit `None`, quantize_lm_head `False`, batch `8`, calibrate `False`
- values in %, +- = lm-eval stderr; delta = preset - float (points)
- `float` keeps Q/K/V and the attention output in fp32, so `delta <preset>` includes the per-(batch, head) INT8 attention quantisation, not only the softmax; `vs float_int8attn` = preset - float_int8attn (same INT8 attention, fp32 softmax) isolates the softmax
- dynamic per-sequence INT8 scales make the INT8-attention presets' continuation log-probs depend on later tokens (not strictly causal)
- paper column: [A] the paper's metric is assumed to be acc_norm for ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)

| task | metric | float | float_int8attn | cimple | delta float_int8attn | delta cimple | vs float_int8attn cimple | paper CIMple | paper delta |
|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | acc | 30.63 +- 1.35 | 29.69 +- 1.34 | 30.20 +- 1.34 | -0.94 | -0.43 | +0.51 |  |  |
| ARC-Challenge | acc_norm | 32.42 +- 1.37 | 32.94 +- 1.37 | 33.36 +- 1.38 | +0.51 | +0.94 | +0.43 | 31.74* | +0.60* |
| OpenBookQA | acc | 24.60 +- 1.93 | 25.20 +- 1.94 | 25.20 +- 1.94 | +0.60 | +0.60 | +0.00 |  |  |
| OpenBookQA | acc_norm | 35.20 +- 2.14 | 36.20 +- 2.15 | 36.60 +- 2.16 | +1.00 | +1.40 | +0.40 | 36.80* | -0.20* |
| Winogrande | acc | 60.22 +- 1.38 | 60.62 +- 1.37 | 59.43 +- 1.38 | +0.39 | -0.79 | -1.18 | 57.22* | -0.32* |

| preset | docs | eval s | docs/s | threads | batch | softmax config |
|---|---|---|---|---|---|---|
| float | 2939 | 4245.3 | 0.692 | 11 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=False, quantize_out=False)` |
| float_int8attn | 2939 | 4620.3 | 0.636 | 8 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |
| cimple | 2939 | 4966.1 | 0.592 | 11 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |

| preset | task | docs | requests | eval s | requests/s | s/doc |
|---|---|---|---|---|---|---|
| float | ARC-Challenge | 1172 | 4687 | 3245.7 | 1.44 | 2.769 |
| float | OpenBookQA | 500 | 2000 | 509.8 | 3.92 | 1.02 |
| float | Winogrande | 1267 | 2534 | 489.8 | 5.17 | 0.387 |
| float_int8attn | ARC-Challenge | 1172 | 4687 | 3506.1 | 1.34 | 2.992 |
| float_int8attn | OpenBookQA | 500 | 2000 | 388.2 | 5.15 | 0.776 |
| float_int8attn | Winogrande | 1267 | 2534 | 726.1 | 3.49 | 0.573 |
| cimple | ARC-Challenge | 1172 | 4687 | 3902.6 | 1.2 | 3.33 |
| cimple | OpenBookQA | 500 | 2000 | 374.6 | 5.34 | 0.749 |
| cimple | Winogrande | 1267 | 2534 | 688.9 | 3.68 | 0.544 |
