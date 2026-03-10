"""
run_task2b_sparse_bound.py — Task 2B: Sparse-corrected triangle bounds.

For each (tau, vocab_size) entry in results/task2_triangle_counts.json,
compute the theoretically correct baseline for a sparse random graph G(n, p_D):

    E[T] = C(n,3) * p_D^3
    Var[T] ≈ C(n,3)*p_D^3*(1-p_D^3) + 3*C(n,2)*(n-2)*p_D^5*(1-p_D)
    Z = (observed - E[T]) / sqrt(Var[T])

Z >> 0 means the corpus has significantly MORE triangles than a random graph
at the same density, indicating non-random structure driving hallucinations.

This is the theoretically correct version of the Goodman comparison (Task 2
compared against a complete-graph bound which was never applicable to a
sparse co-occurrence graph).

Outputs:
    results/task2b_sparse_bound.json
    results/task2b_sparse_triangles.png  — observed vs expected per tau, log-log
    results/task2b_zscore.png            — Z-score vs n per tau

Usage:
    python scripts/run_task2b_sparse_bound.py \\
        --task2-results results/task2_triangle_counts.json \\
        --results-dir results
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

from src.sparse_bounds import analyze_sparse_bound


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_observed_vs_expected(
    results_by_tau: dict,
    output_path: str,
) -> None:
    """
    Log-log plot: observed triangles vs expected_sparse vs n, one line per tau.

    Panel A: triangle counts (observed, expected_sparse) vs n
    Panel B: Z-score vs n
    """
    taus = sorted(results_by_tau.keys())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(taus) - 1, 1)) for i in range(len(taus))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: observed vs expected (log-log) ───────────────────────────────
    ax1 = axes[0]
    for tau, color in zip(taus, colors):
        rows = results_by_tau[tau]
        # Sort by n
        rows_sorted = sorted(rows, key=lambda r: r["n_vertices"])
        ns = [r["n_vertices"] for r in rows_sorted]
        obs = [r["observed_triangles"] for r in rows_sorted]
        exp = [r["expected_sparse"] for r in rows_sorted]

        # Clip to avoid log(0)
        obs_plot = [max(o, 1) for o in obs]
        exp_plot = [max(e, 1e-3) for e in exp]

        ax1.plot(ns, obs_plot, "o-", color=color, label=f"τ={tau} obs", linewidth=1.5,
                 markersize=4)
        ax1.plot(ns, exp_plot, "--", color=color, label=f"τ={tau} exp", linewidth=1,
                 alpha=0.7)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Vocabulary size n", fontsize=12)
    ax1.set_ylabel("Triangle count", fontsize=12)
    ax1.set_title("Observed vs Expected Triangles\n(solid=observed, dashed=G(n,p) expected)",
                  fontsize=11)
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, which="both", alpha=0.3)

    # ── Panel B: Z-score vs n ─────────────────────────────────────────────────
    ax2 = axes[1]
    for tau, color in zip(taus, colors):
        rows = results_by_tau[tau]
        rows_sorted = sorted(rows, key=lambda r: r["n_vertices"])
        ns = [r["n_vertices"] for r in rows_sorted]
        zs = [r["z_score"] for r in rows_sorted]
        ax2.plot(ns, zs, "o-", color=color, label=f"τ={tau}", linewidth=1.5, markersize=4)

    ax2.axhline(y=0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax2.axhline(y=3, color="red", linewidth=1, linestyle="--", alpha=0.7,
                label="Z=3 threshold")
    ax2.axhline(y=-3, color="blue", linewidth=1, linestyle="--", alpha=0.7,
                label="Z=-3 threshold")

    ax2.set_xscale("log")
    ax2.set_xlabel("Vocabulary size n", fontsize=12)
    ax2.set_ylabel("Z-score (obs - E[T]) / σ", fontsize=12)
    ax2.set_title("Z-score vs Vocabulary Size\n(Z>>0: more triangles than random graph)",
                  fontsize=11)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 2B: Sparse-corrected triangle bounds for G(n, p)"
    )
    parser.add_argument(
        "--task2-results",
        default="results/task2_triangle_counts.json",
        help="Path to task2_triangle_counts.json produced by run_task2_triangles.py",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    task2_path = Path(args.task2_results)
    if not task2_path.exists():
        print(f"ERROR: Task 2 results not found at {task2_path}")
        print("Run Task 2 first: python scripts/run_task2_triangles.py ...")
        sys.exit(1)

    with task2_path.open() as fh:
        task2_data = json.load(fh)

    # task2_triangle_counts.json structure: list of experiment dicts OR a dict
    # with "results" key. Handle both formats.
    if isinstance(task2_data, list):
        experiments = task2_data
    elif isinstance(task2_data, dict) and "results" in task2_data:
        experiments = task2_data["results"]
    elif isinstance(task2_data, dict):
        # May have tau keys at top level, flatten
        experiments = []
        for key, val in task2_data.items():
            if isinstance(val, list):
                experiments.extend(val)
            elif isinstance(val, dict) and "observed_triangles" in val:
                experiments.append(val)
    else:
        print("ERROR: Unexpected format in task2_triangle_counts.json")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments from {task2_path}")

    # ── Compute sparse bounds for each experiment ─────────────────────────────
    all_results = []
    results_by_tau: dict = {}

    for exp in experiments:
        tau = float(exp.get("tau", exp.get("threshold", 2.0)))
        vocab_size = int(exp.get("vocab_size", exp.get("n_vertices", exp.get("n", 1000))))
        n = int(exp.get("n_vertices", exp.get("n", vocab_size)))
        p = float(exp.get("edge_density", exp.get("density", 0.0)))
        observed = int(exp.get("observed_triangles", exp.get("triangles", 0)))

        if n < 3 or p <= 0:
            print(f"  Skipping tau={tau}, vocab_size={vocab_size}: n={n}, p={p}")
            continue

        row = analyze_sparse_bound(observed, n, p, tau, vocab_size)
        all_results.append(row)

        if tau not in results_by_tau:
            results_by_tau[tau] = []
        results_by_tau[tau].append(row)

        print(
            f"  tau={tau:.1f}, n={n:6d}, p_D={p:.5f} | "
            f"observed={observed:10,d}, expected={row['expected_sparse']:12.1f}, "
            f"Z={row['z_score']:+.2f}, SER={row['sparse_excess_ratio']:.4f}"
        )

    if not all_results:
        print("No valid experiments to analyze.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_significant_more = sum(1 for r in all_results if r["significantly_more_than_random"])
    n_significant_less = sum(1 for r in all_results if r["significantly_fewer_than_random"])
    n_consistent = sum(1 for r in all_results if r["consistent_with_random"])

    z_values = [r["z_score"] for r in all_results]
    z_mean = float(np.mean(z_values))
    z_median = float(np.median(z_values))
    z_max = float(np.max(z_values))
    z_min = float(np.min(z_values))

    summary = {
        "source": str(task2_path),
        "n_experiments": len(all_results),
        "n_taus": len(results_by_tau),
        "z_score_mean": z_mean,
        "z_score_median": z_median,
        "z_score_max": z_max,
        "z_score_min": z_min,
        "n_significantly_more_than_random": n_significant_more,
        "n_significantly_fewer_than_random": n_significant_less,
        "n_consistent_with_random": n_consistent,
        "interpretation": (
            "STRONG NON-RANDOM STRUCTURE: corpus has significantly more triangles "
            "than G(n,p) at same density. This indicates co-occurrence graph has "
            "genuine clustering not attributable to edge density alone."
            if z_mean > 3
            else (
                "CONSISTENT WITH RANDOM: triangle counts are within expected range "
                "for a random graph G(n,p) at the observed edge density."
                if abs(z_mean) <= 3
                else "FEWER TRIANGLES THAN RANDOM: sub-random clustering structure."
            )
        ),
        "experiments": all_results,
    }

    out_path = results_dir / "task2b_sparse_bound.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved sparse bound results to {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_path = results_dir / "task2b_sparse_triangles.png"
    plot_observed_vs_expected(results_by_tau, str(plot_path))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== TASK 2B SUMMARY ===")
    print(f"Experiments analyzed: {len(all_results)}")
    print(f"Z-score range: [{z_min:.2f}, {z_max:.2f}]  mean={z_mean:.2f}  median={z_median:.2f}")
    print(f"  Z > +3 (more triangles than random): {n_significant_more}/{len(all_results)}")
    print(f"  Z ≈ 0  (consistent with random):     {n_consistent}/{len(all_results)}")
    print(f"  Z < -3 (fewer triangles than random): {n_significant_less}/{len(all_results)}")
    print()
    print(f"Interpretation: {summary['interpretation']}")
    print()
    if z_mean > 3:
        print("RESULT: The PPMI co-occurrence graph has significantly more triangle")
        print("structure than a random graph G(n, p_D) with the same edge density.")
        print("This non-random clustering is a structural property of language that")
        print("may drive hallucinations via the Ramsey mechanism.")
    else:
        print("RESULT: Triangle counts are consistent with a random graph at this density.")
        print("The Ramsey effect may not be the primary driver, or the effect is masked")
        print("by the choice of vocabulary size / PPMI threshold.")


if __name__ == "__main__":
    main()
