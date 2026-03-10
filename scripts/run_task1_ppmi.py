"""
run_task1_ppmi.py — Task 1: Build the PPMI graph and characterize its structure.

Computes the PPMI token co-occurrence graph on WikiText-103 and answers:
    - What is the edge density p_D at each threshold?
    - Does the degree distribution follow a power law?
    - What does the clustering coefficient look like?

Outputs:
    results/task1_ppmi_stats.json
    results/task1_degree_distribution.png

Usage:
    python scripts/run_task1_ppmi.py \
        --corpus data/wikitext103_train.json \
        --vocab-size 10000 \
        --window 5 \
        --tau 1.0 2.0 3.0
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import build_vocabulary
from src.ppmi import (
    build_cooccurrence_matrix,
    compute_ppmi_matrix,
    graph_stats,
    load_ppmi,
    save_ppmi,
    threshold_ppmi,
)


def fit_power_law(degrees: np.ndarray) -> dict:
    """
    Estimate power-law exponent via linear regression on log-log degree distribution.

    Args:
        degrees: Array of vertex degrees.

    Returns:
        Dict with exponent, r_squared, and whether fit looks like power law (r^2 > 0.9).
    """
    degrees = degrees[degrees > 0]
    if len(degrees) < 10:
        return {"exponent": None, "r_squared": None, "is_power_law": False}

    values, counts = np.unique(degrees, return_counts=True)
    log_k = np.log10(values.astype(float))
    log_p = np.log10(counts.astype(float) / counts.sum())

    # Linear regression on log-log
    coeffs = np.polyfit(log_k, log_p, 1)
    exponent = coeffs[0]
    predicted = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "exponent": float(exponent),
        "r_squared": float(r_squared),
        "is_power_law": bool(r_squared > 0.85 and exponent < -1.0),
    }


def plot_degree_distributions(
    degree_arrays: dict,
    output_path: str,
) -> None:
    """
    Plot degree distributions (log-log) for each tau threshold.

    Args:
        degree_arrays: Dict mapping tau -> numpy array of degrees.
        output_path: Where to save the figure.
    """
    fig, axes = plt.subplots(1, len(degree_arrays), figsize=(5 * len(degree_arrays), 4))
    if len(degree_arrays) == 1:
        axes = [axes]

    for ax, (tau, degrees) in zip(axes, sorted(degree_arrays.items())):
        degrees = degrees[degrees > 0]
        values, counts = np.unique(degrees, return_counts=True)
        probs = counts / counts.sum()

        ax.loglog(values, probs, "o", markersize=3, alpha=0.7, label=f"tau={tau}")

        # Fit and overlay power law
        fit = fit_power_law(degrees)
        if fit["exponent"] is not None:
            log_k = np.log10(values.astype(float))
            coeffs = np.polyfit(log_k, np.log10(probs), 1)
            fitted_probs = 10 ** np.polyval(coeffs, log_k)
            ax.loglog(values, fitted_probs, "r--", alpha=0.7,
                      label=f"fit: α={fit['exponent']:.2f}, R²={fit['r_squared']:.2f}")

        ax.set_xlabel("Degree k")
        ax.set_ylabel("P(k)")
        ax.set_title(f"Degree Distribution (τ={tau})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved degree distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Task 1: Build PPMI graph on WikiText-103")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz",
                        help="Cache path for PPMI matrix (reuses if exists)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Build vocabulary
    token2id, id2token = build_vocabulary(
        args.corpus,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )

    # Step 2: Build co-occurrence matrix + PPMI (or load from cache)
    ppmi_cache = Path(args.ppmi_cache)
    if ppmi_cache.exists():
        print(f"Loading cached PPMI matrix from {ppmi_cache}")
        ppmi = load_ppmi(str(ppmi_cache))
    else:
        cooc = build_cooccurrence_matrix(args.corpus, token2id, window=args.window)
        ppmi = compute_ppmi_matrix(cooc)
        ppmi_cache.parent.mkdir(parents=True, exist_ok=True)
        save_ppmi(ppmi, str(ppmi_cache))

    # Step 3: Threshold at each tau and compute stats
    all_stats = {}
    degree_arrays = {}

    for tau in args.tau:
        print(f"\n--- Threshold tau={tau} ---")
        adj = threshold_ppmi(ppmi, tau)
        stats = graph_stats(adj)
        stats["tau"] = tau

        degrees = np.asarray(adj.sum(axis=1)).flatten()
        degree_arrays[tau] = degrees

        # Power-law fit
        power_law = fit_power_law(degrees)
        stats["power_law_fit"] = power_law

        all_stats[str(tau)] = stats
        print(f"  Vertices: {stats['n_vertices']}, Edges: {stats['n_edges']}, "
              f"Density: {stats['edge_density']:.6f}")
        print(f"  Degree mean: {stats['degree_mean']:.2f}, max: {stats['degree_max']}")
        print(f"  Power law: exponent={power_law['exponent']}, "
              f"R²={power_law['r_squared']}, is_power_law={power_law['is_power_law']}")

    # Step 4: Save stats JSON
    stats_path = Path(args.results_dir) / "task1_ppmi_stats.json"
    with stats_path.open("w") as fh:
        json.dump({
            "vocab_size": len(token2id),
            "window": args.window,
            "thresholds": all_stats,
        }, fh, indent=2)
    print(f"\nSaved stats to {stats_path}")

    # Step 5: Plot degree distributions
    plot_path = Path(args.results_dir) / "task1_degree_distribution.png"
    plot_degree_distributions(degree_arrays, str(plot_path))

    # Step 6: Summary
    print("\n=== TASK 1 SUMMARY ===")
    print(f"{'tau':>6} {'vertices':>10} {'edges':>12} {'density':>12} {'power_law':>12} {'R²':>8}")
    print("-" * 65)
    for tau in sorted(args.tau):
        s = all_stats[str(tau)]
        pl = s["power_law_fit"]
        print(f"{tau:>6.1f} {s['n_vertices']:>10,} {s['n_edges']:>12,} "
              f"{s['edge_density']:>12.6f} {str(pl['is_power_law']):>12} "
              f"{(pl['r_squared'] or 0.0):>8.3f}")

    print("\nKey question: Does degree distribution follow a power law?")
    for tau in sorted(args.tau):
        pl = all_stats[str(tau)]["power_law_fit"]
        answer = "YES (use scale-free Ramsey)" if pl["is_power_law"] else "NO (use Rödl-Ruciński G(n,p))"
        print(f"  tau={tau}: {answer}")


if __name__ == "__main__":
    main()
