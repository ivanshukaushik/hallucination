"""
run_task2_triangles.py — Task 2: Count triangles vs Goodman bound.

Counts the actual number of token triples where all three PPMI scores exceed
tau, and compares to the Goodman lower bound.

Key question: At what vocabulary size n does the observed triangle count
cross the Goodman bound? This empirically locates the phase transition n*.

Outputs:
    results/task2_triangle_counts.json
    results/task2_goodman_comparison.png

Usage:
    python scripts/run_task2_triangles.py --tau 1.0 2.0 3.0
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

from src.ppmi import load_ppmi, threshold_ppmi
from src.triangle_counter import (
    analyze_triangles,
    goodman_lower_bound,
    find_phase_transition,
)


def plot_goodman_comparison(results: list, output_path: str) -> None:
    """
    Plot observed triangles vs Goodman lower bound vs vocabulary size.

    Args:
        results: List of dicts from analyze_triangles.
        output_path: Save path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by tau
    taus = sorted(set(r["tau"] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

    ax1 = axes[0]
    ax2 = axes[1]

    for tau, color in zip(taus, colors):
        tau_results = [r for r in results if r["tau"] == tau]
        ns = [r["n_vertices"] for r in tau_results]
        obs = [r["observed_triangles"] for r in tau_results]
        bounds = [r["goodman_lower_bound"] for r in tau_results]
        rers = [r["ramsey_excess_ratio"] for r in tau_results]

        # Plot 1: Observed vs Goodman bound
        ax1.semilogy(ns, obs, "o-", color=color, label=f"Observed (τ={tau})", markersize=5)
        ax1.semilogy(ns, bounds, "--", color=color, alpha=0.6, label=f"Goodman (τ={tau})")

    ax1.set_xlabel("Vocabulary size n")
    ax1.set_ylabel("Number of triangles (log scale)")
    ax1.set_title("Observed Triangles vs Goodman Lower Bound")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ramsey Excess Ratio
    for tau, color in zip(taus, colors):
        tau_results = [r for r in results if r["tau"] == tau]
        ns = [r["n_vertices"] for r in tau_results]
        rers = [r["ramsey_excess_ratio"] for r in tau_results]
        ax2.semilogy(ns, rers, "o-", color=color, label=f"τ={tau}", markersize=5)

    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="RER = 1 (at Goodman floor)")
    ax2.set_xlabel("Vocabulary size n")
    ax2.set_ylabel("Ramsey Excess Ratio (log scale)")
    ax2.set_title("Ramsey Excess Ratio = Observed / Goodman Bound")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Goodman comparison plot to {output_path}")


def run_vocab_size_sweep(
    ppmi_path: str,
    taus: list,
    vocab_sizes: list,
) -> list:
    """
    Run triangle analysis across multiple vocab sizes to find phase transition.

    Subsamples the vocabulary to different sizes and runs triangle analysis
    at each size and each threshold tau.

    Args:
        ppmi_path: Path to cached PPMI matrix npz.
        taus: List of tau thresholds.
        vocab_sizes: List of vocabulary sizes to sweep over.

    Returns:
        List of result dicts from analyze_triangles.
    """
    ppmi = load_ppmi(ppmi_path)
    full_n = ppmi.shape[0]

    all_results = []
    for tau in taus:
        for n in vocab_sizes:
            if n > full_n:
                n = full_n
            # Take top-n tokens (already ordered by frequency)
            ppmi_sub = ppmi[:n, :n]
            adj = threshold_ppmi(ppmi_sub, tau)
            result = analyze_triangles(adj, tau=tau, verbose=False)
            result["vocab_size_used"] = int(n)
            all_results.append(result)
            print(f"tau={tau}, n={n}: triangles={result['observed_triangles']:,}, "
                  f"goodman={result['goodman_lower_bound']:,}, RER={result['ramsey_excess_ratio']:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Task 2: Count triangles vs Goodman bound")
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz")
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 5000, 7500, 10000],
        help="Vocabulary sizes to sweep for phase transition analysis",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    ppmi_path = Path(args.ppmi_cache)
    if not ppmi_path.exists():
        print(f"ERROR: PPMI cache not found at {ppmi_path}")
        print("Run Task 1 first: python scripts/run_task1_ppmi.py ...")
        sys.exit(1)

    print("Running triangle analysis across vocabulary sizes ...")
    print("(This may take several minutes for large vocabularies)\n")

    all_results = run_vocab_size_sweep(str(ppmi_path), args.tau, args.vocab_sizes)

    # Find phase transitions
    for tau in args.tau:
        tau_results = [r for r in all_results if r["tau"] == tau]
        ns = [r["n_vertices"] for r in tau_results]
        densities = [r["edge_density"] for r in tau_results]
        n_star = find_phase_transition(ns, densities, c=1.0)
        print(f"\ntau={tau}: Rödl-Ruciński phase transition n* ≈ {n_star}")

    # Save results
    out_path = Path(args.results_dir) / "task2_triangle_counts.json"
    with out_path.open("w") as fh:
        json.dump({"results": all_results}, fh, indent=2)
    print(f"\nSaved to {out_path}")

    # Plot
    plot_path = Path(args.results_dir) / "task2_goodman_comparison.png"
    plot_goodman_comparison(all_results, str(plot_path))

    # Summary table
    print("\n=== TASK 2 SUMMARY ===")
    print(f"{'tau':>5} {'n':>8} {'observed':>14} {'goodman':>14} {'RER':>10} {'p_D*n':>8}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: (x["tau"], x["n_vertices"])):
        print(f"{r['tau']:>5.1f} {r['n_vertices']:>8,} {r['observed_triangles']:>14,} "
              f"{r['goodman_lower_bound']:>14,} {r['ramsey_excess_ratio']:>10.4f} "
              f"{r['rodl_rucinski_product']:>8.4f}")

    print("\nKey question: Where does observed triangle count cross Goodman bound?")
    print("(RER ≈ 1 means near the combinatorial floor - theoretically interesting)")


if __name__ == "__main__":
    main()
