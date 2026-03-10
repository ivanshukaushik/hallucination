"""
run_task2c_phase_transition.py — Task 2C: Find the sparse-triangle phase transition n*.

Task 2B showed Z >> 0 at ALL swept vocabulary sizes (1000–10 000).  This script
sweeps SMALLER subgraphs to locate the exact n* where the Z-score first reaches
the 3-sigma threshold.

Method
------
The full 10 000-token PPMI matrix is already stored in data/ppmi_matrix.npz.
Rather than rebuilding the matrix from scratch, we extract subgraphs by taking
the top-n tokens sorted by degree (most-connected tokens come first, giving the
densest subgraph for the smallest n).

For each subgraph of size n:
  1. Slice rows/cols to the top-n token indices.
  2. Threshold at tau to get the binary adjacency submatrix.
  3. Compute edge density p_D.
  4. Count observed triangles via trace(A^3)/6.
  5. Compute expected_sparse = C(n,3)*p_D^3 and variance (local-dependence formula).
  6. Z = (observed - expected) / sqrt(variance).

The smallest n where Z >= 3 is recorded as n_star.

Outputs
-------
  results/task2c_phase_transition.json
  results/task2c_phase_transition.png

Usage
-----
    python scripts/run_task2c_phase_transition.py \\
        --ppmi-cache data/ppmi_matrix.npz \\
        --tau 2.0
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppmi import load_ppmi, threshold_ppmi
from src.sparse_bounds import analyze_sparse_bound
from src.triangle_counter import count_triangles_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Subgraph extraction
# ──────────────────────────────────────────────────────────────────────────────

def top_n_subgraph_by_degree(adj_full: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """
    Extract the induced subgraph on the top-n tokens ranked by degree.

    Using degree-ranked tokens gives the densest possible subgraph for a
    given n, which is conservative for detecting the phase transition
    (if Z < 3 here, it will certainly be < 3 for a random n-token subset).

    Args:
        adj_full: Binary adjacency matrix of the full G_D (shape V x V).
        n: Number of top-degree tokens to keep.

    Returns:
        (n x n) binary CSR adjacency submatrix.
    """
    degrees = np.asarray(adj_full.sum(axis=1)).flatten()
    top_idx = np.argsort(degrees)[::-1][:n]
    top_idx_sorted = np.sort(top_idx)  # keep sorted for efficient slicing
    sub = adj_full[top_idx_sorted][:, top_idx_sorted]
    return sub.tocsr()


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_phase_transition(results: list, n_star: int | None, output_path: str) -> None:
    """
    Single-panel plot: Z-score vs n, with Z=3 threshold line and n* annotation.

    Args:
        results: List of per-n result dicts.
        n_star: First n where Z >= 3, or None.
        output_path: Path to save PNG.
    """
    ns = [r["n_vertices"] for r in results]
    zs = [r["z_score"] for r in results]
    sers = [r["sparse_excess_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Z-score vs n ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(ns, zs, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Z-score")
    ax1.axhline(y=3.0, color="red", linewidth=1.5, linestyle="--", label="Z = 3 (3σ threshold)")
    ax1.axhline(y=0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    if n_star is not None:
        ax1.axvline(x=n_star, color="orange", linewidth=1.5, linestyle=":",
                    label=f"n* = {n_star}")
        # Find y value at n_star
        z_at_nstar = next((r["z_score"] for r in results if r["n_vertices"] == n_star), None)
        if z_at_nstar is not None:
            ax1.annotate(
                f"n* = {n_star}\nZ = {z_at_nstar:.1f}",
                xy=(n_star, z_at_nstar),
                xytext=(n_star * 1.1, max(z_at_nstar - 5, 0)),
                arrowprops={"arrowstyle": "->", "color": "orange"},
                fontsize=9, color="orange",
            )

    ax1.set_xlabel("Vocabulary size n (top-degree tokens)", fontsize=12)
    ax1.set_ylabel("Z-score", fontsize=12)
    ax1.set_title("Sparse Phase Transition: Z-score vs n\n"
                  "(Z>3: more triangles than G(n,p) random graph)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel B: Sparse Excess Ratio vs n (log scale) ─────────────────────────
    ax2 = axes[1]
    ax2.plot(ns, sers, "s-", color="#d62728", linewidth=2, markersize=5, label="SER")
    ax2.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.6,
                label="SER = 1 (random baseline)")

    if n_star is not None:
        ax2.axvline(x=n_star, color="orange", linewidth=1.5, linestyle=":",
                    label=f"n* = {n_star}")

    ax2.set_xlabel("Vocabulary size n (top-degree tokens)", fontsize=12)
    ax2.set_ylabel("Sparse Excess Ratio (obs / E[T])", fontsize=12)
    ax2.set_title("Sparse Excess Ratio vs n\n(SER > 1: more triangles than expected)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phase transition plot to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 2C: Find sparse-triangle phase transition n*"
    )
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz",
                        help="Path to PPMI matrix NPZ file (from Task 1)")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="PPMI threshold used for G_D (default 2.0)")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 500, 750, 1000],
        help="Vocabulary sizes to sweep (default: 50 100 150 200 300 500 750 1000)",
    )
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Z-score threshold that defines n* (default 3.0)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ppmi_path = Path(args.ppmi_cache)
    if not ppmi_path.exists():
        print(f"ERROR: PPMI cache not found at {ppmi_path}")
        print("Run Task 1 first: python scripts/run_task1_ppmi.py ...")
        sys.exit(1)

    # Load full PPMI matrix and threshold once
    print(f"Loading PPMI matrix from {ppmi_path} ...")
    ppmi_full = load_ppmi(str(ppmi_path))
    print(f"  Shape: {ppmi_full.shape}, nnz={ppmi_full.nnz:,}")

    print(f"Thresholding at tau={args.tau} ...")
    adj_full = threshold_ppmi(ppmi_full, args.tau)
    V_full = adj_full.shape[0]
    n_edges_full = int(adj_full.nnz) // 2
    print(f"  Full graph: V={V_full}, edges={n_edges_full:,}")

    # Sort vocab_sizes — must not exceed V_full
    vocab_sizes = sorted(v for v in args.vocab_sizes if v <= V_full)
    if not vocab_sizes:
        print(f"ERROR: All requested vocab sizes exceed the matrix size ({V_full}).")
        sys.exit(1)
    if len(vocab_sizes) < len(args.vocab_sizes):
        skipped = [v for v in args.vocab_sizes if v > V_full]
        print(f"WARNING: Skipping vocab sizes > {V_full}: {skipped}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_results = []
    n_star = None

    for n in vocab_sizes:
        sub_adj = top_n_subgraph_by_degree(adj_full, n)

        n_eff = sub_adj.shape[0]
        n_edges = int(sub_adj.nnz) // 2
        max_edges = n_eff * (n_eff - 1) / 2
        p_D = n_edges / max_edges if max_edges > 0 else 0.0

        observed = count_triangles_matrix(sub_adj)

        row = analyze_sparse_bound(
            observed=observed,
            n=n_eff,
            p=p_D,
            tau=args.tau,
            vocab_size=n,
        )
        all_results.append(row)

        z = row["z_score"]
        ser = row["sparse_excess_ratio"]

        print(
            f"  n={n_eff:5d}  p_D={p_D:.5f}  obs={observed:8,d}  "
            f"exp={row['expected_sparse']:10.1f}  Z={z:+8.2f}  SER={ser:.4f}"
        )

        if n_star is None and z >= args.z_threshold:
            n_star = n_eff
            print(f"  *** Phase transition n* = {n_star} (Z={z:.2f} >= {args.z_threshold}) ***")

    # ── Summary ───────────────────────────────────────────────────────────────
    z_values = [r["z_score"] for r in all_results]
    summary = {
        "tau": args.tau,
        "z_threshold": args.z_threshold,
        "n_star": n_star,
        "n_star_interpretation": (
            f"Smallest subgraph (top-degree tokens) where triangle count exceeds "
            f"G(n,p) expectation by {args.z_threshold}σ: n*={n_star}"
            if n_star is not None
            else f"No n* found: Z never reached {args.z_threshold} in swept range"
        ),
        "z_score_min": float(min(z_values)),
        "z_score_max": float(max(z_values)),
        "swept_vocab_sizes": vocab_sizes,
        "experiments": all_results,
    }

    out_path = results_dir / "task2c_phase_transition.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved results to {out_path}")

    plot_path = results_dir / "task2c_phase_transition.png"
    plot_phase_transition(all_results, n_star, str(plot_path))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== TASK 2C SUMMARY ===")
    if n_star is not None:
        print(f"Phase transition n* = {n_star}")
        print(f"  At n={n_star}, the PPMI co-occurrence subgraph first has significantly")
        print(f"  more triangles than expected in G(n,p_D) at the same edge density.")
        print(f"  This marks the onset of non-random Ramsey structure in the corpus.")
    else:
        print(f"No phase transition found in the swept range {vocab_sizes}.")
        print(f"Z-scores ranged from {min(z_values):.2f} to {max(z_values):.2f}.")
        print(f"  If Z < 0 everywhere: try lower tau or extend sweep to larger n.")
        print(f"  If Z > 0 but < 3: structure exists but is weak at these sizes.")


if __name__ == "__main__":
    main()
