# CIMple softmax accuracy study (Sec. V-B, Fig. 11)

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, quant `w8a8`, 0-shot, limit `24`
- values in %, +- = lm-eval stderr; delta = preset - float (points)
- paper column: [A] the paper's metric is assumed to be acc_norm for ARC-C/HellaSwag/OBQA and acc for Winogrande (marked `*`)

| task | metric | float | cimple | cimple_hw8 | delta cimple | delta cimple_hw8 | paper CIMple | paper delta |
|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | acc | 20.83 +- 8.47 | 20.83 +- 8.47 | 16.67 +- 7.77 | +0.00 | -4.17 |  |  |
| ARC-Challenge | acc_norm | 29.17 +- 9.48 | 29.17 +- 9.48 | 12.50 +- 6.90 | +0.00 | -16.67 | 31.74* | +0.60* |
| HellaSwag | acc | 50.00 +- 10.43 | 50.00 +- 10.43 | 33.33 +- 9.83 | +0.00 | -16.67 |  |  |
| HellaSwag | acc_norm | 58.33 +- 10.28 | 54.17 +- 10.39 | 20.83 +- 8.47 | -4.17 | -37.50 | 60.52* | -0.54* |
| OpenBookQA | acc | 8.33 +- 5.76 | 12.50 +- 6.90 | 8.33 +- 5.76 | +4.17 | +0.00 |  |  |
| OpenBookQA | acc_norm | 37.50 +- 10.09 | 37.50 +- 10.09 | 25.00 +- 9.03 | +0.00 | -12.50 | 36.80* | -0.20* |
| Winogrande | acc | 54.17 +- 10.39 | 54.17 +- 10.39 | 41.67 +- 10.28 | +0.00 | -12.50 | 57.22* | -0.32* |

| preset | docs | eval s | docs/s | threads | batch | softmax config |
|---|---|---|---|---|---|---|
| float | 96 | 86.3 | 1.113 | 16 | 8 | `SoftmaxConfig(mode='float', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=False, quantize_out=False)` |
| cimple | 96 | 97.1 | 0.988 | 16 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='full', recip='exact', quantize_qkv=True, quantize_out=True)` |
| cimple_hw8 | 96 | 101.6 | 0.945 | 16 | 8 | `SoftmaxConfig(mode='split_lut', score_bits=8, score_scale='dynamic', zmax='const', exp_precision='hw8', recip='lut', quantize_qkv=True, quantize_out=True)` |

| preset | task | docs | requests | eval s | requests/s | s/doc |
|---|---|---|---|---|---|---|
| float | ARC-Challenge | 24 | 96 | 27.2 | 3.52 | 1.135 |
| float | HellaSwag | 24 | 96 | 25.7 | 3.73 | 1.072 |
| float | OpenBookQA | 24 | 96 | 17.4 | 5.51 | 0.726 |
| float | Winogrande | 24 | 48 | 15.9 | 3.02 | 0.662 |
| cimple | ARC-Challenge | 24 | 96 | 31.5 | 3.05 | 1.312 |
| cimple | HellaSwag | 24 | 96 | 30.1 | 3.19 | 1.254 |
| cimple | OpenBookQA | 24 | 96 | 18.7 | 5.13 | 0.779 |
| cimple | Winogrande | 24 | 48 | 16.9 | 2.85 | 0.703 |
| cimple_hw8 | ARC-Challenge | 24 | 96 | 32.4 | 2.96 | 1.351 |
| cimple_hw8 | HellaSwag | 24 | 96 | 31.6 | 3.04 | 1.317 |
| cimple_hw8 | OpenBookQA | 24 | 96 | 20.3 | 4.73 | 0.846 |
| cimple_hw8 | Winogrande | 24 | 48 | 17.2 | 2.78 | 0.718 |
