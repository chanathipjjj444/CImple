# CIMple softmax accuracy study (Sec. V-B, Fig. 11)

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, quant `w8a8`, 0-shot, limit `250`, quantize_lm_head `False`, batch `8`, calibrate `False`
- values in %, +- = lm-eval stderr; delta = preset - float (points)
- `float` keeps Q/K/V and the attention output in fp32, so `delta <preset>` includes the per-(batch, head) INT8 attention quantisation, not only the softmax; `vs float_int8attn` = preset - float_int8attn (same INT8 attention, fp32 softmax) isolates the softmax
- dynamic per-sequence INT8 scales make the INT8-attention presets' continuation log-probs depend on later tokens (not strictly causal)
- paper column: [A] the paper's metric is assumed to be acc_norm for ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)

| task | metric | float | float_int8attn | cimple_hw8 | cimple_hw8_rowmax | delta float_int8attn | delta cimple_hw8 | delta cimple_hw8_rowmax | vs float_int8attn cimple_hw8 | vs float_int8attn cimple_hw8_rowmax | paper CIMple | paper delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | acc | 29.60 +- 2.89 | 28.00 +- 2.85 | 24.00 +- 2.71 | 28.40 +- 2.86 | -1.60 | -5.60 | -1.20 | -4.00 | +0.40 |  |  |
| ARC-Challenge | acc_norm | 34.40 +- 3.01 | 34.40 +- 3.01 | 25.20 +- 2.75 | 32.40 +- 2.97 | +0.00 | -9.20 | -2.00 | -9.20 | -2.00 | 31.74* | +0.60* |
| HellaSwag | acc | 46.80 +- 3.16 | 46.00 +- 3.16 | 30.00 +- 2.90 | 46.00 +- 3.16 | -0.80 | -16.80 | -0.80 | -16.00 | +0.00 |  |  |
| HellaSwag | acc_norm | 57.60 +- 3.13 | 55.60 +- 3.15 | 32.40 +- 2.97 | 56.00 +- 3.15 | -2.00 | -25.20 | -1.60 | -23.20 | +0.40 | 60.52* | -0.54* |
| OpenBookQA | acc | 22.80 +- 2.66 | 23.60 +- 2.69 | 16.80 +- 2.37 | 22.40 +- 2.64 | +0.80 | -6.00 | -0.40 | -6.80 | -1.20 |  |  |
| OpenBookQA | acc_norm | 37.60 +- 3.07 | 39.60 +- 3.10 | 32.40 +- 2.97 | 36.80 +- 3.06 | +2.00 | -5.20 | -0.80 | -7.20 | -2.80 | 36.80* | -0.20* |
| Winogrande | acc | 59.60 +- 3.11 | 58.40 +- 3.12 | 48.80 +- 3.17 | 58.00 +- 3.13 | -1.20 | -10.80 | -1.60 | -9.60 | -0.40 | 57.22* | -0.32* |

| preset | docs | eval s | docs/s | threads | batch | softmax config |
|---|---|---|---|---|---|---|
| float | 1000 | 3133.4 | 0.319 | 4 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=False, quantize_out=False)` |
| float_int8attn | 1000 | 1290.8 | 0.775 | 8 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |
| cimple_hw8 | 1000 | 3623.3 | 0.276 | 4 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='hw8', recip='lut', quantize_qkv=True, quantize_out=True)` |
| cimple_hw8_rowmax | 1000 | 3682.6 | 0.272 | 4 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='row', exp_precision='hw8', recip='lut', quantize_qkv=True, quantize_out=True)` |

| preset | task | docs | requests | eval s | requests/s | s/doc |
|---|---|---|---|---|---|---|
| float | ARC-Challenge | 250 | 999 | 868.0 | 1.15 | 3.472 |
| float | HellaSwag | 250 | 1000 | 1305.4 | 0.77 | 5.222 |
| float | OpenBookQA | 250 | 1000 | 548.3 | 1.82 | 2.193 |
| float | Winogrande | 250 | 500 | 411.8 | 1.21 | 1.647 |
| float_int8attn | ARC-Challenge | 250 | 999 | 426.6 | 2.34 | 1.706 |
| float_int8attn | HellaSwag | 250 | 1000 | 504.9 | 1.98 | 2.02 |
| float_int8attn | OpenBookQA | 250 | 1000 | 206.0 | 4.85 | 0.824 |
| float_int8attn | Winogrande | 250 | 500 | 153.2 | 3.26 | 0.613 |
| cimple_hw8 | ARC-Challenge | 250 | 999 | 991.4 | 1.01 | 3.965 |
| cimple_hw8 | HellaSwag | 250 | 1000 | 1536.5 | 0.65 | 6.146 |
| cimple_hw8 | OpenBookQA | 250 | 1000 | 671.1 | 1.49 | 2.684 |
| cimple_hw8 | Winogrande | 250 | 500 | 424.4 | 1.18 | 1.697 |
| cimple_hw8_rowmax | ARC-Challenge | 250 | 999 | 1020.0 | 0.98 | 4.08 |
| cimple_hw8_rowmax | HellaSwag | 250 | 1000 | 1562.8 | 0.64 | 6.251 |
| cimple_hw8_rowmax | OpenBookQA | 250 | 1000 | 668.8 | 1.5 | 2.675 |
| cimple_hw8_rowmax | Winogrande | 250 | 500 | 431.1 | 1.16 | 1.724 |
