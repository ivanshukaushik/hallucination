"""
run_task3_rhi.py — Task 3: Test the Ramsey Hallucination Index against TruthfulQA.

Uses GPT-2 to generate answers for TruthfulQA questions, labels hallucinations,
checks for PPMI triangles, and runs statistical tests.

This is the first empirical test of the Ramsey-hallucination connection.
p < 0.05 on the chi-squared test would be a publishable result.

Outputs:
    results/task3_rhi_gpt2.json
    results/task3_contingency_table.png

Usage:
    python scripts/run_task3_rhi.py \
        --truthfulqa data/TruthfulQA.csv \
        --ppmi-cache data/ppmi_matrix.npz \
        --tau 2.0 \
        --device cpu
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

    # Compute RHI
    summary = compute_rhi(records, generated_answers, adj, token2id)
    summary.pop("raw_results", None)  # Don't serialize all raw data to JSON by default

    # Save results
    out_path = Path(args.results_dir) / "task3_rhi_gpt2.json"
    with out_path.open("w") as fh:
        json.dump({
            "tau": args.tau,
            "vocab_size": args.vocab_size,
            "model": "gpt2",
            "dataset": "TruthfulQA",
            **summary,
        }, fh, indent=2)
    print(f"Saved results to {out_path}")

    # Plot
    plot_path = Path(args.results_dir) / "task3_contingency_table.png"
    plot_contingency(summary, str(plot_path))

    # Final interpretation
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
