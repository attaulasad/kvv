# Quantization-Induced Hallucination in Precomputed-KV-Cache RAG

*Does offline KV-cache compression hurt faithfulness more than it hurts factual accuracy?*

## Abstract

Retrieval-augmented generation (RAG) systems can amortize prefill cost by
precomputing and storing the key–value (KV) cache of retrieved passages, then
compressing those caches to save storage and I/O. This work asks whether such
offline KV-cache quantization degrades *faithfulness* (whether the answer is
supported by the retrieved context) faster than it degrades *factual accuracy*
(containment exact-match). Using Qwen2.5-Instruct on the RGB robustness benchmark,
we compare an uncompressed full-context oracle against FP16, INT8, and group-wise
INT4 KV caches across retrieval depths K ∈ {1, 3, 5}. We measure accuracy
(containment-EM, token-F1), faithfulness (Vectara HHEM-2.1-Open and DeBERTa-v3
NLI, with an optional LLM-as-judge anchor), and efficiency (KV-cache size, time to
first token, latency). A Stage-0 baseline-fidelity gate verifies that the lossless
(FP16) condition matches the oracle before any quantized numbers are collected, and
McNemar / regression tests quantify the asymmetry between accuracy and faithfulness
degradation.

## Method summary

The default precompute path is **`standard_causal`**: the retrieved chunks are
concatenated and prefilled with a single stock causal pass, and *that* cache is the
artifact we quantize. This is in-distribution, requires no fine-tuning, and isolates
the effect of storage-level compression. The compression conditions are:

| Condition | Cache | Description |
|---|---|---|
| **C0** | — | Full-context oracle. Raw text context, standard generation. Upper-bound reference. |
| **C1** | FP16 | Prefilled cache round-tripped through the codec at 16-bit (lossless reference; should match C0). |
| **C2** | INT8 | Per-token asymmetric INT8 quantization of the prefilled cache, dequantized to bfloat16. |
| **C3** | INT4 | **Group-wise** symmetric INT4 quantization (**group size 64**, packed two nibbles per byte), dequantized to bfloat16. |

**INT4 is group-wise with a group size of 64.** The 128-dim head vector is split
into contiguous 64-channel groups, each with its own absmax scale. Group-wise
scaling localizes the damage from a single large-magnitude channel to its own group
rather than letting one outlier set the scale for the whole vector — the standard
fix used by KIVI / KVQuant. See [`src/kv_quantization.py`](src/kv_quantization.py).

> Qwen2.5 is bfloat16-native, so all caches are stored and dequantized in bfloat16;
> casting to float16 overflows the KV values and corrupts the logits. The pipeline
> asserts the loaded dtype and aborts on any non-finite logit.

## Install

Requires Python 3.10+ and a CUDA GPU (developed on a single NVIDIA RTX 3090, 24 GB;
peak usage is ~10 GB, so there is comfortable headroom).

```bash
pip install -r requirements.txt          # pins transformers==4.51.3
```

Set the cache/scratch locations (override the defaults as needed):

```bash
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER/kv_quant}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
```

The optional LLM-as-judge stage additionally needs `export ANTHROPIC_API_KEY=...`;
without it that stage is skipped and you still get HHEM + NLI faithfulness scores.

## Reproduce the RGB experiment

The repository is configured out of the box for the headline RGB-only run
(RGB × C0/C1/C2/C3 × K ∈ {1, 3, 5}, bfloat16, `standard_causal`). First gate-check
the baseline, then run the full pipeline:

```bash
# 1. Baseline-fidelity gate: runs C0 + C1 on 50 examples and fails loudly unless
#    C1 ≈ C0 with zero non-finite logits and degeneracy < 2%.
python src/run_experiment.py --config configs/full_experiment.yaml --stages build stage0

# 2. Full headline run (~2–3.5 h on one RTX 3090).
python src/run_experiment.py --config configs/full_experiment.yaml
```

Every setting is read from [`configs/full_experiment.yaml`](configs/full_experiment.yaml)
and can be overridden on the command line (e.g. `--model_name`, `--conditions`,
`--k_values`, `--num_rgb_examples`). The output-directory slug is derived from the
active `--model_name` at runtime. To run the 7B model:

```bash
python src/run_experiment.py --config configs/full_experiment.yaml \
    --model_name Qwen/Qwen2.5-7B-Instruct
```

Inspect the fully resolved config and the generated stage arguments without running
anything:

```bash
python src/run_experiment.py --config configs/full_experiment.yaml --dry_run
```

### Optional secondary datasets

The pipeline can also evaluate NQ-Open and HotpotQA, which are disabled
(`num_examples: 0`) for the headline run. They are kept as optional secondary
baselines: set their `num_examples > 0` in the YAML (HotpotQA loads its own
paragraphs; NQ-Open additionally requires a DPR Wikipedia index via `--wiki_pages`).

## Expected outputs

Results are written under `$SCRATCH_DIR/results/<model_slug>/` and
`$SCRATCH_DIR/analysis/<model_slug>/`:

| Path | Contents |
|---|---|
| `results/<model_slug>/config.json` | Fully resolved run configuration |
| `results/<model_slug>/results_<ts>.jsonl` | Per-example: query, prediction, context, TTFT, latency, KV bytes, per-example HHEM/NLI |
| `results/<model_slug>/summary_<ts>.csv` / `.json` | Main metric table (EM, containment-EM, F1, hallucination rate, entailment, KV size, TTFT, latency) |
| `results/<model_slug>/meta_<ts>.json` | git commit, library versions, seed |
| `analysis/<model_slug>/paper_table.{csv,md,tex}` | Publication master table |
| `analysis/<model_slug>/figure{1,2,3}_data.csv` | Figure data for the accuracy/faithfulness/efficiency plots |
| `analysis/<model_slug>/report.txt` | Human-readable H1/H2/H3 hypothesis verdicts |

A valid run reports `n_nonfinite == 0` everywhere, low `degenerate_rate`, C0/C1 as
the most faithful conditions, and faithfulness degrading from C2 to C3.

The reported hypotheses are: **H1** (asymmetric degradation) — McNemar test on
FP16→INT4 faithfulness flips within the accuracy-preserved subset; **H2**
(multi-chunk amplification) — slope of the paired INT4−FP16 hallucination gap versus
K; **H3** (noise complexity) — slope of that gap versus the retrieved-distractor
ratio on RGB.

## Project structure

```
├── configs/
│   └── full_experiment.yaml    # single source of truth for the experiment
├── src/
│   ├── qwen2.py                # Qwen2 attention variant with raw (un-rotated) key caching
│   ├── kv_quantization.py      # FP16 / INT8 / group-wise INT4 offline KV-cache codec
│   ├── chunk_cache.py          # corpus build + retrieval index (+ optional per-chunk KV caches)
│   ├── evaluate.py             # evaluation loop over conditions × K × datasets
│   ├── metrics.py              # EM, F1, HHEM, DeBERTa-NLI scorers
│   ├── calibrate_metrics.py    # HHEM vs DeBERTa-NLI correlation
│   ├── analyze_results.py      # H1/H2/H3 hypothesis tests + figure CSVs
│   ├── make_paper_tables.py    # publication tables (CSV / Markdown / LaTeX)
│   ├── llm_judge.py            # optional Claude LLM-as-judge faithfulness anchor
│   ├── config.py               # YAML loader (expands ${SCRATCH_DIR}, ${MODEL_SLUG})
│   └── run_experiment.py       # orchestrator — reads YAML + CLI, runs all stages
├── scripts/                    # thin shell wrappers around run_experiment.py stages
├── questions/
│   └── rgb.jsonl               # RGB dataset: query, answer, positive[] + negative[] docs
└── requirements.txt
```

RGB ships its own evidence documents in `questions/rgb.jsonl`; its `positive` and
`negative` passages are ingested into the retrieval corpus automatically.

## Citation

```bibtex
@misc{kvquant_rag,
  title  = {Quantization-Induced Hallucination in Precomputed-KV-Cache RAG:
            Does Offline KV Cache Compression Hurt Faithfulness More Than
            Factual Accuracy?},
  author = {<authors>},
  year   = {2026},
  note   = {Workshop submission}
}
```
