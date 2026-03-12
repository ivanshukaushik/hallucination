"""
run_theory_estimation.py — Estimate information-theoretic hallucination floors.

Loads empirical RHI results and estimates the quantities from src/theory.py:
  - rho: per-token out-of-support probability (from category frequency heuristic)
  - alpha_n: fraction of ambiguous questions (approximated from hallucination rate)
  - epsilon_n: mean uncertainty in ambiguous questions (approximated from triangle density)
  - Theorem-1 floor: R*_n >= (1 - delta_n) / 2
  - Rare-Evidence floor: R*_n >= (1/2)(1-rho)^n
  - Ambiguous-Region floor: R*_n >= epsilon_n * alpha_n

Also produces a sensitivity table over rho x n values.

Inputs (optional — gracefully falls back to synthetic data if absent):
  results/task3_rhi_gpt2_semantic.json   (global RHI statistics)
  results/task3_rhi_by_category.json     (per-category statistics)

Outputs:
  results/theory_estimation.json         (per-category floor estimates)
  results/theory_floor_table.json        (sensitivity table)
  results/theory_floor_table.png         (heatmap visualization)

Usage:
    python scripts/run_theory_estimation.py
    python scripts/run_theory_estimation.py \\
        --rhi-json results/task3_rhi_gpt2_semantic.json \\
        --category-json results/task3_rhi_by_category.json \\
        --context-length 50
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

from src.theory import (
    ambiguous_region_bound,
    compute_delta_n_from_data,
    floor_sensitivity_table,
    hallucination_floor,
    rare_evidence_bound,
    rho_from_category_frequencies,
    tv_distance_upper_bound,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_rhi_json(path: str) -> dict:
    """Load the global RHI JSON (task3_rhi_gpt2_semantic.json)."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return json.load(fh)


def load_category_json(path: str) -> list:
    """
    Load per-category RHI JSON.

    Handles the format written by run_task3_rhi.py where the top-level
    key 'by_category' holds a list of per-category experiment dicts.
    Returns [] if file is absent.
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as fh:
        data = json.load(fh)
    # Support both list-at-root and {'by_category': [...]} formats
    if isinstance(data, list):
        return data
    return data.get("by_category", data.get("experiments", []))


# ──────────────────────────────────────────────────────────────────────────────
# Estimation logic
# ──────────────────────────────────────────────────────────────────────────────

def estimate_per_category(
    category_data: list,
    context_length: int = 50,
) -> list:
    """
    Estimate floor bounds for each category.

    For each category experiment dict we use:
      - n_hallucinated / n_total  →  alpha_n proxy (fraction labelled hallucinated)
      - triangle_density_hallucinated (if present) → epsilon_n proxy
      - category question count for rho heuristic

    Args:
        category_data: List of per-category dicts from the RHI JSON.
        context_length: Assumed GPT-2 context token count (default 50).

    Returns:
        List of per-category result dicts sorted by category name.
    """
    # Build (category, n_questions) for rho estimation
    cat_pairs = []
    for cat in category_data:
        name = cat.get("category", "unknown")
        n_total = cat.get("n_total", cat.get("n", 0))
        cat_pairs.append((name, int(n_total)))

    # Global rho from category distribution
    rho_global = rho_from_category_frequencies(cat_pairs) if cat_pairs else 0.01

    results = []
    for cat in category_data:
        name = cat.get("category", "unknown")
        n_total = int(cat.get("n_total", cat.get("n", 0)))
        n_hallucinated = int(cat.get("n_hallucinated", 0))

        # alpha_n: fraction hallucinated (proxy for fraction ambiguous)
        alpha_n = n_hallucinated / n_total if n_total > 0 else 0.0

        # epsilon_n: proxy from triangle density of hallucinated answers
        # If not available, use alpha_n itself as a rough uncertainty proxy
        epsilon_n = float(cat.get("triangle_density_hallucinated",
                                   cat.get("rhi_2", alpha_n * 0.5)))
        epsilon_n = float(np.clip(epsilon_n, 0.0, 1.0))

        # Category-specific rho: categories with fewer questions are rarer
        n_total_all = sum(n for _, n in cat_pairs)
        rho_cat = 1.0 - (n_total / n_total_all) if n_total_all > 0 else rho_global

        # TV upper bound and Theorem-1 floor
        tv_up = tv_distance_upper_bound(rho_cat, context_length)
        tv_fl = hallucination_floor(min(tv_up, 1.0))

        # Rare-evidence floor
        rare_fl = rare_evidence_bound(rho_cat, context_length)

        # Ambiguous-region floor
        amb_fl = ambiguous_region_bound(alpha_n, epsilon_n)

        # Composite (tightest available lower bound)
        composite = max(tv_fl, rare_fl, amb_fl)

        results.append(
            {
                "category": name,
                "n_total": n_total,
                "n_hallucinated": n_hallucinated,
                "empirical_hallucination_rate": float(alpha_n),
                "rho_category": float(np.clip(rho_cat, 0.0, 1.0)),
                "context_length_n": context_length,
                "alpha_n": float(alpha_n),
                "epsilon_n": float(epsilon_n),
                "tv_upper_bound": float(tv_up),
                "tv_floor_theorem1": float(tv_fl),
                "rare_evidence_floor": float(rare_fl),
                "ambiguous_region_floor": float(amb_fl),
                "composite_floor": float(composite),
                "floor_vs_empirical_gap": float(alpha_n - composite),
            }
        )

    results.sort(key=lambda r: r["category"])
    return results


def estimate_global(
    rhi_data: dict,
    category_data: list,
    context_length: int = 50,
) -> dict:
    """
    Estimate global floor bounds from the full RHI JSON.

    Args:
        rhi_data:      Global RHI dict.
        category_data: Per-category list (used for rho estimation).
        context_length: Context token count assumption.

    Returns:
        Dict of global floor estimates.
    """
    cat_pairs = [
        (cat.get("category", "?"), int(cat.get("n_total", cat.get("n", 0))))
        for cat in category_data
    ]
    rho_global = rho_from_category_frequencies(cat_pairs) if cat_pairs else 0.02

    # Extract global stats from RHI JSON
    n_total = int(rhi_data.get("n_total", rhi_data.get("n_labeled", rhi_data.get("n", 100))))
    n_hallucinated = int(rhi_data.get("n_hallucinated", 0))
    # Guard: n_hallucinated cannot exceed n_total
    n_hallucinated = min(n_hallucinated, n_total)
    hallucination_rate = n_hallucinated / n_total if n_total > 0 else 0.0

    # Build epsilon list: use triangle density per answer if available
    # Otherwise fall back to hallucination_rate * 0.5 for n_hallucinated items
    epsilon_values = rhi_data.get("epsilon_values", None)
    n_ambiguous = n_hallucinated  # treat hallucinated as ambiguous
    if epsilon_values is None:
        # Simple heuristic: ambiguous answers have mean uncertainty 0.5
        epsilon_values = [0.5] * n_ambiguous

    delta_dict = compute_delta_n_from_data(
        n_contexts=max(n_total, 1),
        n_ambiguous=n_ambiguous,
        epsilon_values=epsilon_values[:n_ambiguous],
    )

    tv_up = tv_distance_upper_bound(rho_global, context_length)
    rare_fl = rare_evidence_bound(rho_global, context_length)

    return {
        "n_total": n_total,
        "n_hallucinated": n_hallucinated,
        "empirical_hallucination_rate": float(hallucination_rate),
        "rho_global": float(rho_global),
        "context_length_n": context_length,
        "tv_upper_bound": float(tv_up),
        "tv_floor_theorem1": float(hallucination_floor(min(tv_up, 1.0))),
        "rare_evidence_floor": float(rare_fl),
        **{f"delta_analysis_{k}": v for k, v in delta_dict.items()},
        "composite_floor": float(
            max(
                hallucination_floor(min(tv_up, 1.0)),
                rare_fl,
                delta_dict["composite_floor"],
            )
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_floor_table(
    table: list,
    rho_values: list,
    n_values: list,
    output_path: str,
) -> None:
    """
    Plot heatmaps of tighter_floor and rare_floor over (rho, n) grid.

    Args:
        table:       Output of floor_sensitivity_table().
        rho_values:  Row axis values.
        n_values:    Column axis values.
        output_path: Path to save PNG.
    """
    nr = len(rho_values)
    nc = len(n_values)

    tighter = np.zeros((nr, nc))
    rare = np.zeros((nr, nc))
    tv_fl = np.zeros((nr, nc))

    for row in table:
        ri = rho_values.index(row["rho"])
        ci = n_values.index(row["n"])
        tighter[ri, ci] = row["tighter_floor"]
        rare[ri, ci] = row["rare_floor"]
        tv_fl[ri, ci] = row["tv_floor"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rho_labels = [f"{r:.3f}" for r in rho_values]
    n_labels = [str(n) for n in n_values]

    for ax, data, title in zip(
        axes,
        [tighter, tv_fl, rare],
        ["Tighter Floor max(TV, Rare)", "TV-distance Floor (Theorem 1)", "Rare-Evidence Floor"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.5)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(n_labels, fontsize=9)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rho_labels, fontsize=9)
        ax.set_xlabel("Context length n", fontsize=11)
        ax.set_ylabel("Per-token out-of-support ρ", fontsize=11)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Annotate cells
        for ri in range(nr):
            for ci in range(nc):
                val = data[ri, ci]
                color = "white" if val > 0.35 else "black"
                ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.suptitle(
        "Hallucination Floor Sensitivity: R*_n ≥ f(ρ, n)\n"
        "(Theorem 1 + Rare-Evidence Corollary)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved floor table heatmap to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate IT hallucination floor bounds from empirical RHI data"
    )
    parser.add_argument(
        "--rhi-json",
        default="results/task3_rhi_gpt2_semantic.json",
        help="Global RHI JSON (task3_rhi_gpt2_semantic.json)",
    )
    parser.add_argument(
        "--category-json",
        default="results/task3_rhi_by_category.json",
        help="Per-category RHI JSON (task3_rhi_by_category.json)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=50,
        help="Assumed GPT-2 context token count for floor computations (default 50)",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    rhi_data = load_rhi_json(args.rhi_json)
    category_data = load_category_json(args.category_json)

    if not rhi_data:
        print(f"WARNING: {args.rhi_json} not found — using synthetic placeholder data.")
        rhi_data = {"n_total": 817, "n_hallucinated": 300}

    if not category_data:
        print(f"WARNING: {args.category_json} not found — using synthetic placeholder data.")
        category_data = [
            {"category": "History", "n_total": 100, "n_hallucinated": 40},
            {"category": "Science", "n_total": 80, "n_hallucinated": 30},
            {"category": "Proverbs", "n_total": 60, "n_hallucinated": 35},
            {"category": "Misconceptions", "n_total": 90, "n_hallucinated": 55},
            {"category": "Fiction", "n_total": 70, "n_hallucinated": 25},
        ]

    # ── Global estimates ──────────────────────────────────────────────────────
    print(f"\nEstimating global IT floor bounds (n={args.context_length}) ...")
    global_est = estimate_global(rhi_data, category_data, args.context_length)

    print(f"  Global rho estimate:          {global_est['rho_global']:.4f}")
    print(f"  TV upper bound:               {global_est['tv_upper_bound']:.4f}")
    print(f"  Theorem-1 floor:              {global_est['tv_floor_theorem1']:.4f}")
    print(f"  Rare-evidence floor:          {global_est['rare_evidence_floor']:.4f}")
    print(f"  Composite floor:              {global_est['composite_floor']:.4f}")
    print(f"  Empirical hallucination rate: {global_est['empirical_hallucination_rate']:.4f}")

    # ── Per-category estimates ────────────────────────────────────────────────
    print(f"\nEstimating per-category IT floor bounds ...")
    per_cat = estimate_per_category(category_data, args.context_length)

    print(f"\n{'Category':<30} {'n':>6} {'alpha':>7} {'eps':>7} "
          f"{'amb_fl':>8} {'rare_fl':>8} {'comp_fl':>8} {'gap':>8}")
    print("-" * 92)
    for r in per_cat:
        print(
            f"  {r['category']:<28} {r['n_total']:>6} {r['alpha_n']:>7.3f} "
            f"{r['epsilon_n']:>7.3f} {r['ambiguous_region_floor']:>8.4f} "
            f"{r['rare_evidence_floor']:>8.4f} {r['composite_floor']:>8.4f} "
            f"{r['floor_vs_empirical_gap']:>8.4f}"
        )

    # ── Sensitivity table ─────────────────────────────────────────────────────
    rho_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    n_values = [10, 50, 100, 200, 500, 1000]

    print(f"\nBuilding sensitivity table ({len(rho_values)} rho × {len(n_values)} n) ...")
    table = floor_sensitivity_table(rho_values, n_values)

    print(f"\n{'rho':>8} {'n':>6} {'TV_upper':>10} {'TV_floor':>10} "
          f"{'rare_fl':>10} {'tighter':>10}")
    print("-" * 60)
    for row in table:
        print(
            f"  {row['rho']:>6.3f} {row['n']:>6} {row['tv_upper']:>10.4f} "
            f"{row['tv_floor']:>10.4f} {row['rare_floor']:>10.4f} "
            f"{row['tighter_floor']:>10.4f}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    est_out = {
        "global": global_est,
        "by_category": per_cat,
        "context_length_assumption": args.context_length,
        "notes": (
            "alpha_n proxied from empirical hallucination rate. "
            "epsilon_n proxied from triangle_density_hallucinated (or 0.5 if absent). "
            "rho estimated from category question-count heuristic. "
            "All floors are lower bounds — actual rates may be higher."
        ),
    }

    est_path = results_dir / "theory_estimation.json"
    with est_path.open("w") as fh:
        json.dump(est_out, fh, indent=2)
    print(f"\nSaved per-category estimates to {est_path}")

    table_path = results_dir / "theory_floor_table.json"
    with table_path.open("w") as fh:
        json.dump({"rho_values": rho_values, "n_values": n_values, "table": table}, fh, indent=2)
    print(f"Saved sensitivity table to {table_path}")

    # Plot
    plot_path = results_dir / "theory_floor_table.png"
    plot_floor_table(table, rho_values, n_values, str(plot_path))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== THEORY ESTIMATION SUMMARY ===")
    print(f"Corpus: TruthfulQA (via task3 RHI pipeline)")
    print(f"Context length assumption: n = {args.context_length} tokens")
    print(f"Global rho estimate: {global_est['rho_global']:.4f}")
    print(f"\nTheorem-1 floor (TV-distance): R*_n >= {global_est['tv_floor_theorem1']:.4f}")
    print(f"Rare-evidence floor:           R*_n >= {global_est['rare_evidence_floor']:.4f}")
    print(f"Composite floor:               R*_n >= {global_est['composite_floor']:.4f}")
    print(f"Empirical rate (comparison):          {global_est['empirical_hallucination_rate']:.4f}")
    print(f"\nCAVEAT: Floors are mathematical lower bounds, not predictions.")
    print(f"  rho is estimated from a heuristic, not measured directly.")
    print(f"  Do not interpret composite_floor as an achievable target.")


if __name__ == "__main__":
    main()
