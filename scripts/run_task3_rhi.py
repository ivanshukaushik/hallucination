"""
run_task3_rhi.py — Task 3: Test the Ramsey Hallucination Index against TruthfulQA.

Uses GPT-2 to generate answers for TruthfulQA questions, labels hallucinations,
checks for PPMI triangles, and runs statistical tests.

This is the first empirical test of the Ramsey-hallucination connection.
p < 0.05 on the chi-squared test would be a publishable result.

Outputs (token-overlap labeling, baseline):
    results/task3_rhi_gpt2.json
    results/task3_contingency_table.png

Outputs (semantic labeling, Task B):
    results/task3_rhi_gpt2_semantic.json
    results/task3_contingency_semantic.png

Outputs (per-category analysis, Task C):
    results/task3_rhi_by_category.json
    results/task3_rhi_by_category.png

Usage:
    python scripts/run_task3_rhi.py \\
        --truthfulqa data/TruthfulQA.csv \\
        --ppmi-cache data/ppmi_matrix.npz \\
        --tau 2.0 \\
        --device cpu \\
        --semantic              # enable Task B semantic labeling
        --per-category          # enable Task C per-category breakdown
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import build_vocabulary
from src.ppmi import load_ppmi, threshold_ppmi
from src.rhi import (
    compute_rhi,
    compute_rhi_semantic,
    generate_gpt2_answers,
    load_truthfulqa,
)


def plot_contingency(summary: dict, output_path: str) -> None:
    """
    Visualize the contingency table and RHI.

    Args:
        summary: Output from compute_rhi.
        output_path: Save path.
    """
    ct = summary["contingency_table"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Contingency table heatmap
    ax1 = axes[0]
    table = np.array([
        [ct["hallucinated_with_triangle"], ct["hallucinated_without_triangle"]],
        [ct["correct_with_triangle"], ct["correct_without_triangle"]],
    ])
    im = ax1.imshow(table, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Has PPMI Triangle", "No PPMI Triangle"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Hallucinated", "Correct"])
    ax1.set_title("Contingency Table: PPMI Triangles vs Hallucination")

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(table[i, j]), ha="center", va="center",
                     color="black" if table[i, j] < table.max() * 0.6 else "white",
                     fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax1)

    # Bar chart: triangle rates
    ax2 = axes[1]
    categories = ["Hallucinated\nanswers", "Correct\nanswers"]
    rates = [summary["rhi_empirical"], summary["triangle_rate_correct"]]
    colors = ["#d62728", "#2ca02c"]
    bars = ax2.bar(categories, rates, color=colors, alpha=0.8, edgecolor="black")

    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{rate:.3f}", ha="center", va="bottom", fontsize=12)

    ax2.set_ylabel("Fraction with PPMI Triangle")
    ax2.set_title(
        f"PPMI Triangle Rate by Answer Type\n"
        f"χ²={summary['chi2_statistic']:.3f}, p={summary['p_value']:.4f}"
        f" ({'*' if summary['significant_at_0.05'] else 'n.s.'})"
    )
    ax2.set_ylim(0, max(rates) * 1.3)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved contingency plot to {output_path}")


def plot_rhi_by_category(category_stats: list, output_path: str) -> None:
    """
    Horizontal bar chart of per-category RHI, sorted descending.

    Bars are coloured by significance (p<0.05 = orange, else grey).
    A star (*) marks significant categories.

    Args:
        category_stats: List of per-category dicts from compute_rhi_by_category.
        output_path: Path to save PNG.
    """
    # Sort descending by RHI
    stats = sorted(category_stats, key=lambda r: r["rhi_category"], reverse=True)
    labels = [r["category"] for r in stats]
    rhi_vals = [r["rhi_category"] for r in stats]
    tri_correct = [r["triangle_rate_correct_category"] for r in stats]
    sig = [r.get("significant_at_0.05", False) for r in stats]

    # Approximate error bars using Wilson interval half-width (95 % CI)
    # For a proportion p with n observations: SE ≈ sqrt(p*(1-p)/n)
    def se(p, n):
        if n <= 0:
            return 0.0
        return (p * (1 - p) / n) ** 0.5

    rhi_errors = [
        se(r["rhi_category"], r["n_hallucinated"]) for r in stats
    ]

    y_pos = range(len(labels))
    colors = ["#d62728" if s else "#aec7e8" for s in sig]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
    bars = ax.barh(list(y_pos), rhi_vals, xerr=rhi_errors, align="center",
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 1, "capsize": 2})

    # Overlay triangle_rate_correct as hollow markers
    ax.scatter(tri_correct, list(y_pos), marker="|", s=100, color="black",
               zorder=5, label="Triangle rate (correct)")

    # Stars for significant categories
    for i, (s, v) in enumerate(zip(sig, rhi_vals)):
        if s:
            ax.text(v + rhi_errors[i] + 0.005, i, "★", va="center", fontsize=9,
                    color="#d62728")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RHI (fraction hallucinated with PPMI triangle)", fontsize=11)
    ax.set_title(
        "Per-Category RHI\n"
        "(red=p<0.05, ★=significant, | =triangle rate in correct answers)",
        fontsize=11,
    )
    ax.axvline(x=0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(0, min(1.1, max(rhi_vals) * 1.3 + 0.1))
    ax.grid(True, axis="x", alpha=0.3)

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="p < 0.05"),
        Patch(facecolor="#aec7e8", label="not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-category plot to {output_path}")


def compute_rhi_by_category(raw_results: list, min_n: int = 20) -> list:
    """
    Group RHI raw results by TruthfulQA category and compute per-category stats.

    For categories with >= min_n labeled examples, also runs a chi-squared test.

    Args:
        raw_results: List of per-question dicts from compute_rhi / compute_rhi_semantic.
                     Each dict must have keys: category, hallucinated, has_ppmi_triangle.
        min_n: Minimum number of labeled examples to run chi-squared (default 20).

    Returns:
        List of per-category stat dicts, sorted by rhi_category descending.
    """
    from scipy.stats import chi2_contingency
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in raw_results:
        cat = r.get("category", "Unknown") or "Unknown"
        groups[cat].append(r)

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
            "skipped_chi2": n_total < min_n,
        }

        if n_total >= min_n and n_hallucinated > 0 and n_correct > 0:
            contingency = np.array([[tp, fp], [fn, tn]])
            # Guard against all-zero rows/cols
            if contingency.min() >= 0 and contingency.sum() > 0:
                try:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    entry["chi2_statistic"] = float(chi2)
                    entry["p_value"] = float(p)
                    entry["significant_at_0.05"] = bool(p < 0.05)
                except Exception:
                    pass

        category_stats.append(entry)

    category_stats.sort(key=lambda r: r["rhi_category"], reverse=True)
    return category_stats


def main():
    parser = argparse.ArgumentParser(description="Task 3: Compute empirical RHI on TruthfulQA")
    parser.add_argument("--truthfulqa", default="data/TruthfulQA.csv")
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz")
    parser.add_argument("--corpus", default="data/wikitext103_train.json",
                        help="Corpus (needed to reconstruct vocabulary)")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--answers-cache", default="data/gpt2_answers.json",
                        help="Cache generated answers (expensive to regenerate)")
    parser.add_argument("--results-dir", default="results")
    # Task B
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic similarity labeling (Task B)")
    parser.add_argument("--embeddings-cache", default="data/truthfulqa_embeddings.npz",
                        help="Path to cache sentence-transformer embeddings")
    parser.add_argument("--sim-threshold", type=float, default=0.1,
                        help="Cosine similarity margin for semantic labeling (default 0.1)")
    parser.add_argument(
        "--min-sim-correct",
        type=float,
        default=0.0,
        help=(
            "Minimum cosine similarity to correct answers required to assign a "
            "'correct' label (default 0.0). Set e.g. 0.3 to avoid labelling "
            "marginally-closer-to-correct answers as correct, reducing the "
            "length/fluency confound in GPT-2 continuations."
        ),
    )
    # Task C
    parser.add_argument("--per-category", action="store_true",
                        help="Compute per-category RHI breakdown (Task C)")
    parser.add_argument("--min-category-n", type=int, default=20,
                        help="Min labeled examples per category for chi-squared (default 20)")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Load TruthfulQA
    records = load_truthfulqa(args.truthfulqa)
    if args.max_questions:
        records = records[:args.max_questions]
        print(f"Limited to {len(records)} questions (--max-questions)")

    # Generate or load cached answers
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

    # Load PPMI matrix and build adjacency graph
    ppmi_path = Path(args.ppmi_cache)
    if not ppmi_path.exists():
        print(f"ERROR: PPMI cache not found at {ppmi_path}")
        print("Run Task 1 first: python scripts/run_task1_ppmi.py ...")
        sys.exit(1)

    print(f"Loading PPMI matrix and thresholding at tau={args.tau} ...")
    ppmi = load_ppmi(str(ppmi_path))
    adj = threshold_ppmi(ppmi, args.tau)

    # Rebuild vocabulary (needed for token -> ID mapping)
    if not Path(args.corpus).exists():
        print(f"WARNING: Corpus not found at {args.corpus}. Cannot map answer tokens to vocab.")
        print("RHI computation will show 0 triangles. Run Task 1 with --corpus first.")
        token2id = {}
    else:
        token2id, _ = build_vocabulary(args.corpus, vocab_size=args.vocab_size)

    # ── Baseline RHI (token-overlap labeling) — always run ───────────────────
    summary = compute_rhi(records, generated_answers, adj, token2id)
    raw_results_baseline = summary.pop("raw_results", [])

    # Attach category to baseline raw results from records
    for r, rec in zip(raw_results_baseline, records[:len(raw_results_baseline)]):
        if "category" not in r:
            r["category"] = rec.get("category", "")

    out_path = Path(args.results_dir) / "task3_rhi_gpt2.json"
    with out_path.open("w") as fh:
        json.dump({
            "tau": args.tau,
            "vocab_size": args.vocab_size,
            "model": "gpt2",
            "dataset": "TruthfulQA",
            **summary,
        }, fh, indent=2)
    print(f"Saved baseline results to {out_path}")

    plot_path = Path(args.results_dir) / "task3_contingency_table.png"
    plot_contingency(summary, str(plot_path))

    # ── Task B: semantic labeling ─────────────────────────────────────────────
    raw_results_for_category = raw_results_baseline  # default: use baseline

    if args.semantic:
        print("\n── Task B: Semantic labeling ──")
        semantic_summary = compute_rhi_semantic(
            records,
            generated_answers,
            adj,
            token2id,
            embeddings_cache=args.embeddings_cache,
            threshold=args.sim_threshold,
            min_sim_correct=args.min_sim_correct,
        )
        raw_results_semantic = semantic_summary.pop("raw_results", [])
        raw_results_for_category = raw_results_semantic  # Task C will use semantic labels

        sem_out = Path(args.results_dir) / "task3_rhi_gpt2_semantic.json"
        with sem_out.open("w") as fh:
            json.dump({
                "tau": args.tau,
                "vocab_size": args.vocab_size,
                "model": "gpt2",
                "dataset": "TruthfulQA",
                **semantic_summary,
            }, fh, indent=2)
        print(f"Saved semantic results to {sem_out}")

        sem_plot = Path(args.results_dir) / "task3_contingency_semantic.png"
        plot_contingency(semantic_summary, str(sem_plot))

        print(f"\n── Semantic vs Baseline comparison ──")
        print(f"  Baseline   RHI={summary['rhi_empirical']:.4f}  p={summary['p_value']:.6f}")
        print(f"  Semantic   RHI={semantic_summary['rhi_empirical']:.4f}  p={semantic_summary['p_value']:.6f}")

    # ── Task C: per-category analysis ─────────────────────────────────────────
    if args.per_category:
        print("\n── Task C: Per-category RHI ──")
        if not raw_results_for_category:
            print("WARNING: No raw results available for per-category breakdown. "
                  "Ensure records have category field in TruthfulQA CSV.")
        else:
            category_stats = compute_rhi_by_category(
                raw_results_for_category, min_n=args.min_category_n
            )

            cat_out = Path(args.results_dir) / "task3_rhi_by_category.json"
            with cat_out.open("w") as fh:
                json.dump({
                    "tau": args.tau,
                    "vocab_size": args.vocab_size,
                    "labeling": "semantic" if args.semantic else "token_overlap",
                    "min_category_n": args.min_category_n,
                    "n_categories": len(category_stats),
                    "categories": category_stats,
                }, fh, indent=2)
            print(f"Saved per-category results to {cat_out}")

            cat_plot = Path(args.results_dir) / "task3_rhi_by_category.png"
            plot_rhi_by_category(category_stats, str(cat_plot))

            # Print top/bottom categories
            sig_cats = [c for c in category_stats if c.get("significant_at_0.05")]
            print(f"\n  Categories with p<0.05: {len(sig_cats)}/{len(category_stats)}")
            if sig_cats:
                for c in sig_cats:
                    print(f"    ★ {c['category']}: RHI={c['rhi_category']:.3f} "
                          f"p={c['p_value']:.4f} n={c['n_total']}")
            print(f"\n  Top 5 categories by RHI:")
            for c in category_stats[:5]:
                p_str = f"p={c['p_value']:.4f}" if c["p_value"] is not None else "n<20"
                print(f"    {c['category']}: RHI={c['rhi_category']:.3f} ({p_str}) n={c['n_total']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== TASK 3 SUMMARY ===")
    print(f"RHI_empirical = {summary['rhi_empirical']:.4f}")
    print(f"  (fraction of hallucinated answers containing a PPMI triangle)")
    print(f"Triangle rate in correct answers = {summary['triangle_rate_correct']:.4f}")
    print(f"Chi-squared statistic: {summary['chi2_statistic']:.4f}")
    print(f"p-value: {summary['p_value']:.6f}")

    if summary["significant_at_0.05"]:
        print("\nRESULT: SIGNIFICANT (p < 0.05)")
        print("There is a statistically significant association between")
        print("PPMI triangle membership and hallucination in GPT-2/TruthfulQA.")
        print("This supports the Ramsey-hallucination connection.")
    else:
        print("\nRESULT: NOT SIGNIFICANT (p >= 0.05)")
        print("No statistically significant association found at this threshold.")
        print("Consider: different tau, larger vocab, or a different hallucination labeling method.")


if __name__ == "__main__":
    main()
