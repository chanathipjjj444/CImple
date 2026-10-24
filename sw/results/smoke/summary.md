# CIMple softmax accuracy study (Sec. V-B, Fig. 11)

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, quant `w8a8`, 0-shot, limit `16`
- values in %, +- = lm-eval stderr; delta = preset - float (points)
- paper column: [A] the paper's metric is assumed to be acc_norm for ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)

| task | metric | float | cimple | delta cimple | paper CIMple | paper delta |
|---|---|---|---|---|---|---|
| ARC-Challenge | acc | 12.50 +- 8.54 | 12.50 +- 8.54 | +0.00 |  |  |
| ARC-Challenge | acc_norm | 18.75 +- 10.08 | 18.75 +- 10.08 | +0.00 | 31.74* | +0.60* |
| Winogrande | acc | 62.50 +- 12.50 | 62.50 +- 12.50 | +0.00 | 57.22* | -0.32* |

| preset | docs | eval s | docs/s | threads | batch | softmax config |
|---|---|---|---|---|---|---|
| float | 32 | 29.8 | 1.073 | 16 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=False, quantize_out=False)` |
| cimple | 32 | 30.8 | 1.04 | 16 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |
