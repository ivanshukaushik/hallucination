"""
run_task3_nonlocal_rhi.py — Task 3 (Nonlocal): Test the nonlocal RHI on TruthfulQA.

This script is the theoretically-tightened version of run_task3_rhi.py.
It replaces the bare "does the answer contain any PPMI triangle?" criterion with
the NONLOCAL PPMI triangle criterion:

    A triple (t_i, t_j, t_k) is a nonlocal triangle if:
      1. All three pairwise PPMI values > tau  (globally associated), AND
      2. At least one pair has local co-occurrence count < local_threshold
         within any 5-token window in the training corpus  (locally ungrounded).

This operationalises "statistically real but causally weak" and avoids the
"spurious by construction" overclaim of the zero-co-occurrence version.

A positive and significant nonlocal RHI provides stronger evidence for the
Ramsey mechanism than the baseline RHI: hallucinated answers disproportionately
contain token triples that are globally associated in the PPMI graph yet rarely
appear in close textual proximity — exactly the type of "forced" co-occurrence
that Ramsey-theoretic arguments predict.

Outputs:
    results/task3_rhi_nonlocal.json
    results/task3_contingency_nonlocal.png
    results/task3_rhi_by_category_nonlocal.json   (if --per-category)
    results/task3_rhi_by_category_nonlocal.png    (if --per-category)

Usage:
    venv/bin/python scripts/run_task3_nonlocal_rhi.py \\
        --truthfulqa data/TruthfulQA.csv \\
        --ppmi-cache data/ppmi_matrix.npz \\
        --corpus data/wikitext103_train.json \\
        --local-cooc-cache data/local_cooc_matrix.npz \\
        --local-threshold 10 \\
        --semantic --sim-threshold 0.0 --min-sim-correct 0.3 \\
        --per-category --min-category-n 8
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import build_local_cooccurrence_matrix, build_vocabulary
from src.ppmi import load_ppmi, threshold_ppmi
from src.rhi import (
    compute_rhi_nonlocal,
    count_nonlocal_ppmi_triangles,
    extract_content_tokens,
    is_hallucinated_semantic,
    load_or_build_embeddings,
    load_truthfulqa,
    generate_gpt2_answers,
    _compute_triangle_density_stats,
)


# ──────────────────────────────────────────────────────────────────────────────
# Semantic-labelled nonlocal RHI
# ──────────────────────────────────────────────────────────────────────────────

def compute_rhi_nonlocal_semantic(
    records: list,
    generated_answers: list,
    adj,
    token2id: dict,
    local_cooc,
    local_threshold: int = 10,
    embeddings_cache: str = "data/truthfulqa_embeddings.npz",
    threshold: float = 0.0,
    min_sim_correct: float = 0.0,
) -> dict:
    """
    Nonlocal RHI with semantic similarity labeling.

    Combines the nonlocal triangle criterion (is_nonlocal_triangle) with
    sentence-transformer-based hallucination labeling (is_hallucinated_semantic).

    Args:
        records: TruthfulQA records.
        generated_answers: GPT-2 answers (same length as records).
        adj: Binary sparse adjacency matrix of G_D.
        token2id: Vocabulary mapping.
        local_cooc: Sparse integer co-occurrence matrix.
        local_threshold: Co-occurrence threshold for "locally ungrounded" (default 10).
        embeddings_cache: Path to cached sentence-transformer embeddings.
        threshold: Cosine-sim margin for semantic labeling (default 0.0).
        min_sim_correct: Minimum sim to correct answers for a "correct" label.

    Returns:
        Summary dict (same shape as compute_rhi_nonlocal, plus labeling metadata).
    """
    from tqdm import tqdm

    gen_embs, correct_embs, incorrect_embs = load_or_build_embeddings(
        records, generated_answers, cache_path=embeddings_cache
    )

    results = []
    for i, (rec, answer) in enumerate(tqdm(
        zip(records, generated_answers),
        total=len(records),
        desc="Computing nonlocal RHI (semantic)",
    )):
        hallucinated = is_hallucinated_semantic(
            gen_embs[i], correct_embs[i], incorrect_embs[i], threshold=threshold
        )
        if hallucinated is None:
            continue
        if not hallucinated:
            if float(np.dot(gen_embs[i], correct_embs[i])) < min_sim_correct:
                continue

        token_ids = extract_content_tokens(answer, token2id)
        ids_unique = list(set(token_ids))
        k = len(ids_unique)

        n_triangles = count_nonlocal_ppmi_triangles(ids_unique, adj, local_cooc, local_threshold)
        has_triangle = n_triangles > 0
        max_triangles = k * (k - 1) * (k - 2) // 6
        triangle_density = n_triangles / max(1, max_triangles)

        results.append({
            "question": rec["question"],
            "generated": answer,
            "category": rec.get("category", ""),
            "hallucinated": hallucinated,
            "has_ppmi_triangle": has_triangle,
            "n_content_tokens": k,
            "n_triangles": n_triangles,
            "triangle_density": float(triangle_density),
            "sim_correct": float(np.dot(gen_embs[i], correct_embs[i])),
            "sim_incorrect": float(np.dot(gen_embs[i], incorrect_embs[i])),
        })

    tp = sum(1 for r in results if r["hallucinated"] and r["has_ppmi_triangle"])
    fp = sum(1 for r in results if r["hallucinated"] and not r["has_ppmi_triangle"])
    tn = sum(1 for r in results if not r["hallucinated"] and not r["has_ppmi_triangle"])
    fn = sum(1 for r in results if not r["hallucinated"] and r["has_ppmi_triangle"])

    contingency = np.array([[tp, fp], [fn, tn]])
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)

    n_hallucinated = tp + fp
    n_correct = fn + tn
    rhi_empirical = tp / n_hallucinated if n_hallucinated > 0 else 0.0
    triangle_rate_correct = fn / n_correct if n_correct > 0 else 0.0
    density_stats = _compute_triangle_density_stats(results)

    summary = {
        "metric": "nonlocal_ppmi_triangle",
        "labeling_method": "semantic",
        "local_threshold": local_threshold,
        "sim_threshold": threshold,
        "min_sim_correct": min_sim_correct,
        "n_labeled": len(results),
        "n_hallucinated": int(n_hallucinated),
        "n_correct": int(n_correct),
        "contingency_table": {
            "hallucinated_with_triangle": int(tp),
            "hallucinated_without_triangle": int(fp),
            "correct_without_triangle": int(tn),
            "correct_with_triangle": int(fn),
        },
        "rhi_empirical": float(rhi_empirical),
        "triangle_rate_correct": float(triangle_rate_correct),
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
        **density_stats,
        "raw_results": results,
    }

    print(f"\nNonlocal RHI Results (semantic, local_threshold={local_threshold}):")
    print(f"  Labeled: {len(results)} ({n_hallucinated} hallucinated, {n_correct} correct)")
    print(f"  RHI_empirical (nonlocal): {rhi_empirical:.4f}")
    print(f"  Triangle rate in correct (nonlocal): {triangle_rate_correct:.4f}")
    print(f"  Chi-squared: {chi2_stat:.4f}, p-value: {p_value:.6f}")
    print(f"  Significant at p<0.05: {p_value < 0.05}")
    print(f"  Mean nonlocal density (hall): {density_stats['mean_triangle_density_hallucinated']:.6f}")
    print(f"  Mean nonlocal density (corr): {density_stats['mean_triangle_density_correct']:.6f}")
    if density_stats["density_t_stat"] is not None:
        print(f"  t-stat: {density_stats['density_t_stat']:.4f}, p (one-tailed): {density_stats['density_p_value']:.6f}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Per-category analysis (same logic as run_task3_rhi.py, with BH+Bonferroni)
# ──────────────────────────────────────────────────────────────────────────────

def compute_nonlocal_rhi_by_category(raw_results: list, min_n: int = 20) -> list:
    """Per-category nonlocal RHI with Bonferroni and BH FDR corrections."""
    groups: dict = defaultdict(list)
    for r in raw_results:
        cat = r.get("category", "Unknown") or "Unknown"
        groups[cat].append(r)

    alpha = 0.05
    category_stats = []

    for cat, rows in groups.items():
        tp = sum(1 for r in rows if r["hallucinated"] and r["has_ppmi_triangle"])
        fp = sum(1 for r in rows if r["hallucinated"] and not r["has_ppmi_triangle"])
        tn = sum(1 for r in rows if not r["hallucinated"] and not r["has_ppmi_triangle"])
        fn = sum(1 for r in rows if not r["hallucinated"] and r["has_ppmi_triangle"])

        n_hallucinated = tp + fp
        n_correct = fn + tn
        n_total = n_hallucinated + n_correct

        rhi_cat = tp / n_hallucinated if n_hallucinated > 0 else 0.0
        tri_correct_cat = fn / n_correct if n_correct > 0 else 0.0

        entry = {
            "category": cat,
            "n_total": int(n_total),
            "n_hallucinated": int(n_hallucinated),
            "n_correct": int(n_correct),
            "rhi_category": float(rhi_cat),
            "triangle_rate_correct_category": float(tri_correct_cat),
            "chi2_statistic": None,
            "p_value": None,
            "significant_at_0.05": None,
            "p_bonferroni": None,
            "bonferroni_significant_at_0.05": None,
            "bh_significant_at_0.05": None,
            "skipped_chi2": n_total < min_n,
        }

        if n_total >= min_n and n_hallucinated > 0 and n_correct > 0:
            contingency = np.array([[tp, fp], [fn, tn]])
            if contingency.min() >= 0 and contingency.sum() > 0:
                try:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    entry["chi2_statistic"] = float(chi2)
                    entry["p_value"] = float(p)
                    entry["significant_at_0.05"] = bool(p < alpha)
                except Exception:
                    pass

        category_stats.append(entry)

    tested = [e for e in category_stats if e["p_value"] is not None]
    m = len(tested)
    if m > 0:
        for e in tested:
            p_bonf = min(1.0, e["p_value"] * m)
            e["p_bonferroni"] = float(p_bonf)
            e["bonferroni_significant_at_0.05"] = bool(p_bonf < alpha)

        sorted_tested = sorted(tested, key=lambda e: e["p_value"])
        bh_cutoff_idx = -1
        for k, e in enumerate(sorted_tested, start=1):
            if e["p_value"] <= (k / m) * alpha:
                bh_cutoff_idx = k - 1
        for k, e in enumerate(sorted_tested):
            e["bh_significant_at_0.05"] = bool(k <= bh_cutoff_idx)

        n_bonf = sum(1 for e in tested if e.get("bonferroni_significant_at_0.05"))
        n_bh = sum(1 for e in tested if e.get("bh_significant_at_0.05"))
        print(f"\n  Multiple comparisons ({m} categories tested):")
        print(f"    After Bonferroni correction: {n_bonf} significant")
        print(f"    After BH FDR correction:     {n_bh} significant")

    category_stats.sort(key=lambda r: r["rhi_category"], reverse=True)
    return category_stats


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (shared with run_task3_rhi.py, local copy to keep scripts independent)
# ──────────────────────────────────────────────────────────────────────────────

def plot_contingency(summary: dict, output_path: str) -> None:
    ct = summary["contingency_table"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    table = np.array([
        [ct["hallucinated_with_triangle"], ct["hallucinated_without_triangle"]],
        [ct["correct_with_triangle"], ct["correct_without_triangle"]],
    ])
    im = ax1.imshow(table, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Has Nonlocal Triangle", "No Nonlocal Triangle"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Hallucinated", "Correct"])
    ax1.set_title("Contingency Table: Nonlocal PPMI Triangles vs Hallucination")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(table[i, j]), ha="center", va="center",
                     color="black" if table[i, j] < table.max() * 0.6 else "white",
                     fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    ax2 = axes[1]
    categories = ["Hallucinated\nanswers", "Correct\nanswers"]
    rates = [summary["rhi_empirical"], summary["triangle_rate_correct"]]
    colors = ["#d62728", "#2ca02c"]
    bars = ax2.bar(categories, rates, color=colors, alpha=0.8, edgecolor="black")
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{rate:.3f}", ha="center", va="bottom", fontsize=12)
    ax2.set_ylabel("Fraction with Nonlocal PPMI Triangle")
    ax2.set_title(
        f"Nonlocal PPMI Triangle Rate\n"
        f"χ²={summary['chi2_statistic']:.3f}, p={summary['p_value']:.4f}"
        f" ({'*' if summary['significant_at_0.05'] else 'n.s.'})"
    )
    ax2.set_ylim(0, max(rates) * 1.3 + 0.05)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved contingency plot to {output_path}")


def plot_category_bars(category_stats: list, output_path: str) -> None:
    stats = sorted(category_stats, key=lambda r: r["rhi_category"], reverse=True)
    labels = [r["category"] for r in stats]
    rhi_vals = [r["rhi_category"] for r in stats]
    tri_correct = [r["triangle_rate_correct_category"] for r in stats]

    def se(p, n):
        return (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0

    rhi_errors = [se(r["rhi_category"], r["n_hallucinated"]) for r in stats]
    bh_sig = [r.get("bh_significant_at_0.05", False) or False for r in stats]
    colors = ["#d62728" if s else "#aec7e8" for s in bh_sig]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
    ax.barh(list(range(len(labels))), rhi_vals, xerr=rhi_errors, align="center",
            color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
            error_kw={"elinewidth": 1, "capsize": 2})
    ax.scatter(tri_correct, list(range(len(labels))), marker="|", s=100,
               color="black", zorder=5)
    for i, (s, v, err) in enumerate(zip(bh_sig, rhi_vals, rhi_errors)):
        if s:
            ax.text(v + err + 0.005, i, "★", va="center", fontsize=9, color="#d62728")

    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Nonlocal RHI (fraction hallucinated with nonlocal PPMI triangle)", fontsize=11)
    ax.set_title(
        "Per-Category Nonlocal RHI\n"
        "(red=BH FDR p<0.05, ★=significant, | =rate in correct answers)",
        fontsize=11,
    )
    ax.axvline(x=0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    if rhi_vals:
        ax.set_xlim(0, min(1.1, max(rhi_vals) * 1.3 + 0.1))
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d62728", label="BH FDR p < 0.05"),
        Patch(facecolor="#aec7e8", label="not significant"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-category nonlocal RHI plot to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 3 (Nonlocal): Test nonlocal PPMI triangle RHI on TruthfulQA"
    )
    parser.add_argument("--truthfulqa", default="data/TruthfulQA.csv")
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz")
    parser.add_argument("--corpus", default="data/wikitext103_train.json")
    parser.add_argument(
        "--local-cooc-cache",
        default="data/local_cooc_matrix.npz",
        help="Path to load/save local co-occurrence matrix NPZ cache",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--local-threshold",
        type=int,
        default=10,
        help=(
            "Local co-occurrence count below which a pair is 'locally ungrounded'. "
            "A triangle needs at least one such pair to be counted as nonlocal. "
            "Default 10; 0 = 'spurious by construction' (too strict). "
            "Higher values include more triangles."
        ),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--answers-cache", default="data/gpt2_answers.json")
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic similarity labeling")
    parser.add_argument("--embeddings-cache", default="data/truthfulqa_embeddings.npz")
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    parser.add_argument("--min-sim-correct", type=float, default=0.0)
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument("--min-category-n", type=int, default=20)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    records = load_truthfulqa(args.truthfulqa)
    if args.max_questions:
        records = records[:args.max_questions]

    answers_cache = Path(args.answers_cache)
    if answers_cache.exists():
        print(f"Loading cached GPT-2 answers from {answers_cache}")
        with answers_cache.open() as fh:
            generated_answers = json.load(fh)
        generated_answers = generated_answers[:len(records)]
    else:
        questions = [r["question"] for r in records]
        generated_answers = generate_gpt2_answers(
            questions, max_new_tokens=100, device=args.device
        )
        answers_cache.parent.mkdir(parents=True, exist_ok=True)
        with answers_cache.open("w") as fh:
            json.dump(generated_answers, fh, indent=2)
        print(f"Cached generated answers to {answers_cache}")

    # ── Load PPMI + vocabulary ─────────────────────────────────────────────────
    ppmi_path = Path(args.ppmi_cache)
    if not ppmi_path.exists():
        print(f"ERROR: PPMI cache not found at {ppmi_path}")
        sys.exit(1)
    print(f"Loading PPMI matrix and thresholding at tau={args.tau} ...")
    ppmi = load_ppmi(str(ppmi_path))
    adj = threshold_ppmi(ppmi, args.tau)

    if not Path(args.corpus).exists():
        print(f"ERROR: Corpus not found at {args.corpus}. Required for vocab + local co-occurrence.")
        sys.exit(1)
    token2id, _ = build_vocabulary(args.corpus, vocab_size=args.vocab_size)

    # ── Build / load local co-occurrence matrix ────────────────────────────────
    # Honour the --local-cooc-cache argument as the cache path, but also pass the
    # default cache name (v{vocab}_w5) so build_local_cooccurrence_matrix can find
    # an existing cache even if it was built under the default name.
    cooc_cache_path = Path(args.local_cooc_cache)
    # Determine cache_dir so the function can resolve its standard filename
    cache_dir = str(cooc_cache_path.parent)
    local_cooc = build_local_cooccurrence_matrix(
        corpus_path=args.corpus,
        token2id=token2id,
        vocab_size=args.vocab_size,
        window_size=5,
        cache_dir=cache_dir,
    )

    # ── Run nonlocal RHI ──────────────────────────────────────────────────────
    if args.semantic:
        print("\n── Nonlocal RHI with semantic labeling ──")
        summary = compute_rhi_nonlocal_semantic(
            records=records,
            generated_answers=generated_answers,
            adj=adj,
            token2id=token2id,
            local_cooc=local_cooc,
            local_threshold=args.local_threshold,
            embeddings_cache=args.embeddings_cache,
            threshold=args.sim_threshold,
            min_sim_correct=args.min_sim_correct,
        )
    else:
        summary = compute_rhi_nonlocal(
            records=records,
            generated_answers=generated_answers,
            adj=adj,
            token2id=token2id,
            local_cooc=local_cooc,
            local_threshold=args.local_threshold,
        )

    raw_results = summary.pop("raw_results", [])

    out_path = results_dir / "task3_rhi_nonlocal.json"
    with out_path.open("w") as fh:
        json.dump({
            "tau": args.tau,
            "vocab_size": args.vocab_size,
            "model": "gpt2",
            "dataset": "TruthfulQA",
            **summary,
        }, fh, indent=2)
    print(f"Saved nonlocal RHI results to {out_path}")

    plot_contingency(summary, str(results_dir / "task3_contingency_nonlocal.png"))

    # ── Per-category ──────────────────────────────────────────────────────────
    if args.per_category and raw_results:
        print("\n── Per-category nonlocal RHI ──")
        category_stats = compute_nonlocal_rhi_by_category(raw_results, min_n=args.min_category_n)

        cat_out = results_dir / "task3_rhi_by_category_nonlocal.json"
        with cat_out.open("w") as fh:
            json.dump({
                "tau": args.tau,
                "vocab_size": args.vocab_size,
                "local_threshold": args.local_threshold,
                "labeling": "semantic" if args.semantic else "token_overlap",
                "min_category_n": args.min_category_n,
                "n_categories": len(category_stats),
                "categories": category_stats,
            }, fh, indent=2)
        print(f"Saved per-category nonlocal results to {cat_out}")

        plot_category_bars(category_stats, str(results_dir / "task3_rhi_by_category_nonlocal.png"))

        sig_cats = [c for c in category_stats if c.get("bh_significant_at_0.05")]
        print(f"\n  Categories with BH FDR p<0.05: {len(sig_cats)}/{len(category_stats)}")
        if sig_cats:
            for c in sig_cats:
                print(f"    ★ {c['category']}: RHI={c['rhi_category']:.3f} "
                      f"p={c['p_value']:.4f} n={c['n_total']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== TASK 3 NONLOCAL SUMMARY ===")
    print(f"Metric: nonlocal PPMI triangle (local_threshold={args.local_threshold})")
    print(f"  A triangle must have at least one pair with < {args.local_threshold} "
          f"co-occurrences in a 5-token window.")
    print(f"RHI_nonlocal = {summary['rhi_empirical']:.4f}")
    print(f"Nonlocal triangle rate (correct) = {summary['triangle_rate_correct']:.4f}")
    print(f"Chi-squared: {summary['chi2_statistic']:.4f}, p-value: {summary['p_value']:.6f}")

    if summary["significant_at_0.05"]:
        print("\nRESULT: SIGNIFICANT (p < 0.05)")
        print("Hallucinated answers contain significantly more nonlocal PPMI triangles.")
        print("This supports the Ramsey mechanism: globally associated but locally")
        print("ungrounded token triples are over-represented in hallucinated outputs.")
    else:
        print("\nRESULT: NOT SIGNIFICANT (p >= 0.05)")
        print("No significant difference in nonlocal triangle rates.")
        print("Consider adjusting --local-threshold or --sim-threshold.")


if __name__ == "__main__":
    main()
