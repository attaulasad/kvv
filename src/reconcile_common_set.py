"""Reconcile an already-collected run onto the COMMON 4-way example set (Bug 3).

The faithfulness numbers in a summary written by an OLD evaluate.py are biased:
the C0 Oracle was scored over its own (larger) non-refusal set while C1–C3 used
the smaller paired set, so within one (dataset, K) cell the four conditions had
DIFFERENT denominators (e.g. HotpotQA K=5: C0 n=157 vs C1–C3 n=145). evaluate.py
is now fixed at the root, but re-running it needs a GPU. This post-hoc tool gives
the IDENTICAL correction on CPU, using the per-record HHEM/NLI flags that are
already stored in the results JSONL — no model inference required.

For each (dataset, K) it computes the common set = queries that are present,
non-refusal AND HHEM-scored in EVERY condition, then recomputes per condition:
    n_total, n_hall, hallucination_rate, entailment_score, n_ctx_over_512, paired_n
over that single shared set. EM / F1 / contain_EM / refusal_rate / timing / KV
size are left untouched (they are not part of the faithfulness denominator and so
are not affected by Bug 3).

Outputs:
  --output_summary   corrected summary JSON (feed to analyze + tables)
  --output_jsonl     records restricted to the common set (feed to analyze so H1/
                     H2/figures use the same 4-way set); llm_* fields preserved.

Usage:
  python src/reconcile_common_set.py \
      --summary_json results/.../summary_XXXX.json \
      --results_jsonl results/.../llm_judge/results_judged.jsonl \
      --output_summary results/.../summary_XXXX_reconciled.json \
      --output_jsonl   results/.../results_XXXX_common.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict


def _load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _ctx_over_512(context: str, limit_tokens: int = 512) -> bool:
    # Mirrors metrics.estimate_token_len's tokenizer-free fallback (~4 chars/token).
    # The NLI tokenizer used at eval time is unavailable offline; this estimate is
    # applied identically to every condition, so the per-cell counts stay equal
    # across C0–C3 (which is the actual Bug-3 requirement).
    return (len(context or "") // 4) > limit_tokens


def main():
    ap = argparse.ArgumentParser(description="Reconcile a run onto the common 4-way set (Bug 3)")
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--results_jsonl", required=True,
                    help="Per-example JSONL with hhem_hallucinated / nli_entailment "
                         "(the judged JSONL is fine — llm_* fields are preserved).")
    ap.add_argument("--output_summary", required=True)
    ap.add_argument("--output_jsonl", required=True)
    args = ap.parse_args()

    with open(args.summary_json, encoding="utf-8") as f:
        summary = json.load(f)
    records = _load_jsonl(args.results_jsonl)

    # Index: (dataset, k, condition) -> {query: record}
    by_cell = defaultdict(dict)
    for r in records:
        by_cell[(r["dataset"], r["k"], r["condition"])][r["query"]] = r

    # Conditions present per (dataset, k), taken from the summary so we honour
    # exactly the conditions that were evaluated.
    conds_by_dk = defaultdict(set)
    for row in summary:
        conds_by_dk[(row["dataset"], row["k"])].add(row["condition"])

    # Common set per (dataset, k): queries non-refusal + HHEM-scored in ALL conds.
    common_by_dk = {}
    for (ds, k), conds in conds_by_dk.items():
        cond_list = sorted(conds)
        # candidate queries = those scored in the first condition
        sets = []
        for c in cond_list:
            qmap = by_cell.get((ds, k, c), {})
            ok = {
                q for q, rec in qmap.items()
                if not rec.get("is_refusal") and rec.get("hhem_hallucinated") is not None
            }
            sets.append(ok)
        common = set.intersection(*sets) if sets else set()
        common_by_dk[(ds, k)] = common
        print(f"[reconcile] {ds} K={k}: common 4-way set = {len(common)} "
              f"(conds={cond_list})")

    # ── Recompute faithfulness fields per summary row over the common set ──
    new_summary = []
    for row in summary:
        ds, k, cond = row["dataset"], row["k"], row["condition"]
        common = common_by_dk.get((ds, k), set())
        qmap = by_cell.get((ds, k, cond), {})
        recs = [qmap[q] for q in common if q in qmap]

        flags = [bool(r["hhem_hallucinated"]) for r in recs
                 if r.get("hhem_hallucinated") is not None]
        ents = [float(r["nli_entailment"]) for r in recs
                if r.get("nli_entailment") is not None]
        n_total = len(flags)
        n_hall = sum(flags)
        n_ctx = sum(1 for r in recs if _ctx_over_512(r.get("context", "")))

        new_row = dict(row)
        old_hall = row.get("hallucination_rate")
        old_n = row.get("n_total")
        new_row["n_total"] = n_total
        new_row["n_hall"] = n_hall
        new_row["n_ctx_over_512"] = n_ctx
        new_row["paired_n"] = len(common)
        new_row["hallucination_rate"] = round(n_hall / n_total, 4) if n_total else "N/A"
        new_row["entailment_score"] = round(sum(ents) / len(ents), 4) if ents else "N/A"
        new_row["reconciled_common_set"] = True
        new_summary.append(new_row)
        print(f"[reconcile] {ds} K={k} {cond}: "
              f"hall {old_hall} (n={old_n}) -> {new_row['hallucination_rate']} (n={n_total})")

    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(new_summary, f, indent=2)
    print(f"[reconcile] corrected summary -> {args.output_summary}")

    # ── Write records restricted to the common set (preserve llm_* fields) ──
    n_kept = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            common = common_by_dk.get((r["dataset"], r["k"]), set())
            if r["query"] in common and r["condition"] in conds_by_dk[(r["dataset"], r["k"])]:
                f.write(json.dumps(r) + "\n")
                n_kept += 1
    print(f"[reconcile] common-set records ({n_kept}) -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
