# Precomputed-KV-Cache RAG — Quantization Faithfulness — Main Results

_model_: `Qwen/Qwen2.5-7B-Instruct` · _wiki_pages_: 0 · _datasets_: hotpotqa · _K_: [1, 3, 5] · _conditions_: ['C0', 'C1', 'C2', 'C3']

| Dataset | K | Condition | ContainEM (prim.) | EM | F1 (secondary) | Hallucination | Entailment | Refusal | NonFinite | Degen. | KV Size (MB) | TTFT (s) | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | 1 | Full-Context Oracle | 0.2400 | 0.1100 | 0.1862 | 0.3806 | 0.3180 | 0.5100 | 0 | 0.0000 | 0.000 | 0.8516 | 0.8516 |
| hotpotqa | 1 | FP16 Precomputed-KV | 0.2500 | 0.1133 | 0.1879 | 0.3731 | 0.3237 | 0.5133 | 0 | 0.0000 | 11.539 | 0.0668 | 0.9510 |
| hotpotqa | 1 | INT8 Precomputed-KV | 0.2533 | 0.1167 | 0.1923 | 0.3881 | 0.3328 | 0.5133 | 0 | 0.0000 | 6.130 | 0.0666 | 0.9396 |
| hotpotqa | 1 | INT4 Precomputed-KV | 0.2700 | 0.0433 | 0.1413 | 0.4179 | 0.4989 | 0.3167 | 0 | 0.0167 | 3.245 | 0.0668 | 1.7032 |
| hotpotqa | 3 | Full-Context Oracle | 0.4433 | 0.1800 | 0.3171 | 0.2100 | 0.2980 | 0.2633 | 0 | 0.0000 | 0.000 | 1.0580 | 1.0580 |
| hotpotqa | 3 | FP16 Precomputed-KV | 0.4433 | 0.1833 | 0.3228 | 0.2200 | 0.3196 | 0.2533 | 0 | 0.0000 | 27.135 | 0.0675 | 1.0522 |
| hotpotqa | 3 | INT8 Precomputed-KV | 0.4367 | 0.1767 | 0.3133 | 0.2350 | 0.3035 | 0.2467 | 0 | 0.0000 | 14.416 | 0.0673 | 1.0345 |
| hotpotqa | 3 | INT4 Precomputed-KV | 0.3533 | 0.0667 | 0.1614 | 0.2800 | 0.3370 | 0.2033 | 0 | 0.0267 | 7.632 | 0.0673 | 1.8307 |
| hotpotqa | 5 | Full-Context Oracle | 0.4800 | 0.2067 | 0.3468 | 0.1963 | 0.2417 | 0.2467 | 0 | 0.0000 | 0.000 | 1.1548 | 1.1548 |
| hotpotqa | 5 | FP16 Precomputed-KV | 0.5100 | 0.2200 | 0.3555 | 0.1822 | 0.2571 | 0.2167 | 0 | 0.0000 | 43.829 | 0.0677 | 1.1467 |
| hotpotqa | 5 | INT8 Precomputed-KV | 0.4867 | 0.2100 | 0.3482 | 0.2103 | 0.2346 | 0.2200 | 0 | 0.0000 | 23.284 | 0.0565 | 0.8431 |
| hotpotqa | 5 | INT4 Precomputed-KV | 0.3233 | 0.0533 | 0.1361 | 0.4299 | 0.2326 | 0.0967 | 0 | 0.0300 | 12.327 | 0.0505 | 1.5427 |
