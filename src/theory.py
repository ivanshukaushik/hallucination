"""
theory.py — Information-theoretic floor bounds on LLM hallucination rates.

This module implements the formal quantities from the theoretical analysis in:
  "Ramsey-Theoretic Bounds on LLM Hallucination Rates"

Overview of Results
--------------------
**Theorem 1 (Hallucination Floor from TV Distance)**

  Let P_n be the true distribution over factual answers given n context tokens,
  and Q_n be the model's output distribution.  Then the Bayes risk of any binary
  correct/hallucinated classifier is lower-bounded:

      R*_n  ≥  (1 − δ_n) / 2,   where δ_n = TV(P_n, Q_n) ≤ 2·(1−(1−ρ)^n)

  Here ρ ∈ (0,1) is the per-token marginal probability that a novel context
  token falls outside the training support.  The bound is tight when the
  model's uncertainty is concentrated on a single ambiguous region.

**Rare-Evidence Corollary**

  In a sparse-evidence regime (few training examples per factual entity):

      R*_n  ≥  (1/2)·(1−ρ)^n

  This follows from the complementary bound: when (1−ρ)^n is large (few novel
  tokens, ample coverage), the TV distance is small and the floor is low.

**Ambiguous-Region Bound**

  Let α_n = fraction of questions whose correct answer lies in an "ambiguous
  region" (model assigns non-trivial probability to multiple answers), and
  ε_n = average uncertainty within those ambiguous questions.  Then:

      R*_n  ≥  ε_n · α_n

Safe vs. Unsafe Claims
-----------------------
SAFE (directly supported by these bounds):
  - R*_n ≥ (1−δ_n)/2 is a mathematical consequence of Fano's inequality.
  - The floor rises as TV distance δ_n grows.
  - A corpus with rare evidence (large (1−ρ)^n) has a higher floor.

UNSAFE (not directly supported here):
  - That empirical hallucination rates equal or approach R*_n — the bounds
    are floors, not predictors of actual rates.
  - That the Ramsey graph structure *causes* hallucinations — it is a
    structural correlate, not a causal mechanism.
  - That ρ can be measured directly — it must be estimated from proxy data.

All functions are pure Python (math + numpy only, no ML dependencies).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core floor-bound functions
# ──────────────────────────────────────────────────────────────────────────────

def hallucination_floor(delta_n: float) -> float:
    """
    Bayes-risk lower bound from TV distance (Theorem 1).

    R*_n >= (1 - delta_n) / 2

    For a model with total-variation distance delta_n from the true answer
    distribution, no binary correct/hallucinated classifier can achieve error
    below this floor.

    Args:
        delta_n: TV(P_n, Q_n) in [0, 1].  Values outside [0,1] are clamped.

    Returns:
        Lower bound on Bayes error rate, in [0, 0.5].
    """
    delta_n = float(np.clip(delta_n, 0.0, 1.0))
    return (1.0 - delta_n) / 2.0


def tv_distance_upper_bound(rho: float, n: int) -> float:
    """
    Upper bound on TV(P_n, Q_n) when each context token has marginal
    out-of-support probability rho.

    delta_n  <=  2 * (1 - (1 - rho)^n)

    Derived from a union-bound argument: any of the n context tokens may
    be outside the model's training distribution, and each such token
    contributes at most rho to the TV gap.

    Args:
        rho: Per-token marginal out-of-support probability, in [0, 1].
        n:   Number of context tokens (positive integer).

    Returns:
        Upper bound on TV distance, in [0, 2].  (TV can be at most 1 but
        the formula can exceed 1 for large n; callers should interpret
        values > 1 as "TV is effectively 1".)
    """
    rho = float(np.clip(rho, 0.0, 1.0))
    n = max(1, int(n))
    return 2.0 * (1.0 - (1.0 - rho) ** n)


def rare_evidence_bound(rho: float, n: int) -> float:
    """
    Hallucination floor in the rare-evidence (sparse training) regime.

    R*_n >= (1/2) * (1 - rho)^n

    When (1−ρ)^n is large (most context tokens ARE in-support), the
    TV distance to P_n is small, but by the complementary argument the
    model must hedge between many plausible completions, giving a
    floor proportional to (1−ρ)^n.

    Args:
        rho: Per-token out-of-support probability, in [0, 1].
        n:   Number of context tokens (positive integer).

    Returns:
        Lower bound on hallucination rate, in [0, 0.5].
    """
    rho = float(np.clip(rho, 0.0, 1.0))
    n = max(1, int(n))
    return 0.5 * (1.0 - rho) ** n


def ambiguous_region_bound(alpha_n: float, epsilon_n: float) -> float:
    """
    Hallucination floor from the fraction of ambiguous questions.

    R*_n >= epsilon_n * alpha_n

    Args:
        alpha_n:   Fraction of questions in the ambiguous region, in [0, 1].
        epsilon_n: Mean uncertainty (e.g. entropy / log|vocab|, or 1 − max_prob)
                   within ambiguous questions, in [0, 1].

    Returns:
        Lower bound on hallucination rate.
    """
    alpha_n = float(np.clip(alpha_n, 0.0, 1.0))
    epsilon_n = float(np.clip(epsilon_n, 0.0, 1.0))
    return epsilon_n * alpha_n


# ──────────────────────────────────────────────────────────────────────────────
# Empirical estimation helpers
# ──────────────────────────────────────────────────────────────────────────────

def rho_from_category_frequencies(
    category_question_pairs: Sequence[Tuple[str, int]],
) -> float:
    """
    Estimate the average per-token out-of-support probability rho from
    per-category question counts.

    The estimator treats less-frequently-asked categories as having higher
    out-of-support probability: categories with fewer questions are likely
    to have sparser training evidence.

    Specifically, let n_c = number of questions in category c, and
    N = total questions.  Then rho_c = 1 - n_c / N (heuristic: fraction
    of the question distribution NOT represented by this category).
    The returned rho is weighted by n_c / N.

    Args:
        category_question_pairs: Sequence of (category_name, n_questions).
                                  Must have at least one element.

    Returns:
        Weighted-average rho estimate in [0, 1].
    """
    if not category_question_pairs:
        raise ValueError("category_question_pairs must be non-empty")

    total = sum(n for _, n in category_question_pairs)
    if total == 0:
        raise ValueError("Total question count is zero")

    weighted_rho = 0.0
    for _cat, n_c in category_question_pairs:
        w = n_c / total
        rho_c = 1.0 - n_c / total  # heuristic: how "rare" is this category
        weighted_rho += w * rho_c

    return float(np.clip(weighted_rho, 0.0, 1.0))


def compute_delta_n_from_data(
    n_contexts: int,
    n_ambiguous: int,
    epsilon_values: Sequence[float],
) -> Dict[str, float]:
    """
    Estimate δ_n, α_n, ε_n, and the composite hallucination floor from data.

    Args:
        n_contexts:    Total number of evaluation questions/contexts.
        n_ambiguous:   Number of questions classified as "ambiguous" (model
                       assigns meaningful probability mass to multiple answers).
        epsilon_values: Per-ambiguous-question uncertainty estimates in [0,1]
                        (e.g. 1 − max_softmax_prob, or normalised entropy).
                        Length must equal n_ambiguous.

    Returns:
        Dict with keys:
            alpha_n          — fraction ambiguous
            epsilon_n        — mean uncertainty among ambiguous questions
            ambiguous_floor  — ambiguous_region_bound(alpha_n, epsilon_n)
            delta_n_approx   — approximation: 2 * alpha_n * epsilon_n (proxy)
            tv_floor         — hallucination_floor(delta_n_approx)
            composite_floor  — max(tv_floor, ambiguous_floor)
    """
    if n_contexts <= 0:
        raise ValueError("n_contexts must be positive")
    if n_ambiguous < 0 or n_ambiguous > n_contexts:
        raise ValueError("n_ambiguous must be in [0, n_contexts]")

    eps_arr = np.array(epsilon_values, dtype=float)
    if eps_arr.size != n_ambiguous:
        raise ValueError(
            f"len(epsilon_values)={eps_arr.size} != n_ambiguous={n_ambiguous}"
        )

    alpha_n = n_ambiguous / n_contexts
    epsilon_n = float(eps_arr.mean()) if eps_arr.size > 0 else 0.0
    epsilon_n = float(np.clip(epsilon_n, 0.0, 1.0))

    amb_floor = ambiguous_region_bound(alpha_n, epsilon_n)
    delta_approx = float(np.clip(2.0 * alpha_n * epsilon_n, 0.0, 1.0))
    tv_floor = hallucination_floor(delta_approx)
    composite = max(tv_floor, amb_floor)

    return {
        "alpha_n": float(alpha_n),
        "epsilon_n": float(epsilon_n),
        "ambiguous_floor": float(amb_floor),
        "delta_n_approx": float(delta_approx),
        "tv_floor": float(tv_floor),
        "composite_floor": float(composite),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity table
# ──────────────────────────────────────────────────────────────────────────────

def floor_sensitivity_table(
    rho_values: Sequence[float],
    n_values: Sequence[int],
) -> List[Dict]:
    """
    Compute a sensitivity table of hallucination floor bounds across (rho, n).

    For each (rho, n) pair this computes:
      - tv_upper: tv_distance_upper_bound(rho, n)
      - tv_floor: hallucination_floor(tv_upper)   [from Theorem 1]
      - rare_floor: rare_evidence_bound(rho, n)   [Rare-Evidence Corollary]
      - tighter_floor: max(tv_floor, rare_floor)

    Args:
        rho_values: Sequence of per-token out-of-support probabilities.
        n_values:   Sequence of context lengths (positive integers).

    Returns:
        List of dicts, one per (rho, n) combination, with keys:
            rho, n, tv_upper, tv_floor, rare_floor, tighter_floor.
        Sorted by (rho, n).
    """
    rows = []
    for rho in rho_values:
        for n in n_values:
            tv_up = tv_distance_upper_bound(rho, n)
            tv_fl = hallucination_floor(min(tv_up, 1.0))
            rare_fl = rare_evidence_bound(rho, n)
            rows.append(
                {
                    "rho": float(rho),
                    "n": int(n),
                    "tv_upper": float(tv_up),
                    "tv_floor": float(tv_fl),
                    "rare_floor": float(rare_fl),
                    "tighter_floor": float(max(tv_fl, rare_fl)),
                }
            )
    rows.sort(key=lambda r: (r["rho"], r["n"]))
    return rows
