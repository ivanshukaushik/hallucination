"""
sparse_bounds.py — Sparse-corrected triangle bounds for G(n, p) random graphs.

IMPORTANT — three distinct threshold concepts
=============================================

This module works with three mathematically distinct thresholds.  They are
related but NOT interchangeable.  Conflating them is a common source of error
in Ramsey-flavoured hallucination arguments.

(a) TRIANGLE APPEARANCE THRESHOLD
    p*_appear ≈ n^{-1}   (more precisely: (6/(n(n-1)(n-2)))^{1/3} ≈ 6^{1/3}/n)
    Below this edge probability, G(n,p) almost surely contains NO triangles.
    Above it, the expected number of triangles E[T] = C(n,3)*p^3 grows.
    Reference: Bollobás (2001), Random Graphs, §3.

(b) RAMSEY COLORING THRESHOLD (Rödl-Ruciński 1993)
    p*_Ramsey ≈ n^{-1/2}
    Above this edge probability, G(n,p) almost surely has the property that
    EVERY 2-coloring of its edges contains a monochromatic triangle.
    This is the threshold that directly motivates Ramsey-theoretic hallucination
    arguments — it is far sparser than the complete-graph Goodman bound.
    Reference: Rödl & Ruciński (1993). Lower bounds on probability thresholds
    for Ramsey properties. Contemporary Mathematics 144, 317–346.

(c) GRAPH-STATISTICAL EXCESS (what this module measures)
    The Z-score test here measures whether the observed triangle count in
    the PPMI graph significantly exceeds E[T] = C(n,3)*p_D^3 for a random
    graph G(n, p_D) with the same edge density p_D.
    This is a graph-statistical result — it characterises the PPMI graph as
    having MORE clustering than an Erdős-Rényi random graph at the same density,
    which is interpretable as non-random semantic structure in the corpus.

    *** The Z-score test is NOT directly equivalent to a Ramsey coloring
    theorem. ***  It is inspired by Ramsey-style inevitability arguments: if
    real text graphs have significantly more triangles than random graphs, there
    is a structural basis for Ramsey-type hallucination mechanisms.  But
    empirically exceeding G(n,p) expectation is a weaker and more testable
    claim than "every 2-coloring must contain a monochromatic triangle."

Summary of thresholds for reference:
    p*_appear  ≈  n^{-1}      (triangles start appearing)
    p*_Ramsey  ≈  n^{-1/2}    (Ramsey coloring threshold)
    p_D (observed PPMI graph)  (graph-statistical regime; usually >> p*_appear)

References:
    Bollobás, B. (2001). Random Graphs. Cambridge University Press.
    Rödl & Ruciński (1993). Lower bounds on probability thresholds for Ramsey
        properties. Contemporary Mathematics 144, 317–346.
"""

from math import comb, sqrt
from typing import Tuple


def triangle_appearance_threshold(n: int) -> float:
    """
    Approximate edge probability at which E[triangles] = 1 in G(n, p).

    Setting E[T] = C(n,3)*p^3 = 1 and solving for p:
        p*_appear = (6 / (n*(n-1)*(n-2)))^{1/3}  ≈  (6/n^3)^{1/3}  =  6^{1/3}/n

    Below this threshold, triangles are vanishingly rare in G(n,p).
    Above it, E[T] grows and triangles appear with high probability.

    Args:
        n: Number of vertices.

    Returns:
        Approximate appearance threshold p*_appear (float).
    """
    if n < 3:
        return 1.0
    denom = n * (n - 1) * (n - 2)
    return (6.0 / denom) ** (1.0 / 3.0)


def ramsey_coloring_threshold(n: int) -> float:
    """
    Approximate Rödl-Ruciński (1993) threshold for the Ramsey coloring property.

    Above this edge probability, G(n, p) almost surely has the property that
    every 2-coloring of its edges contains a monochromatic triangle (K_3).
    This is the threshold that directly motivates Ramsey-theoretic arguments
    about LLM hallucinations — it is the density at which "Ramsey inevitability"
    kicks in for the co-occurrence graph.

    The threshold scales as p*_Ramsey ~ n^{-1/2}.  The constant factor is known
    only up to order of magnitude; we use c = 1.0 here as an order-of-magnitude
    estimate.

    IMPORTANT: This threshold concerns 2-COLORINGS of ALL edges of G(n,p) —
    a property of the full graph under adversarial coloring.  It is distinct from
    (a) the appearance threshold (does G(n,p) contain any triangle at all?) and
    (c) the graph-statistical excess (does the observed graph have more triangles
    than Erdős-Rényi expectation?).  The paper measures (c), not (b).

    Args:
        n: Number of vertices.

    Returns:
        Order-of-magnitude estimate of p*_Ramsey = 1.0 / sqrt(n).
    """
    if n < 1:
        return 1.0
    return 1.0 / sqrt(n)


def graph_statistical_regime(n: int, p: float) -> dict:
    """
    Summarise where (n, p) sits relative to the three threshold concepts.

    Used to provide honest framing in results: distinguishing the paper's
    graph-statistical claim from stronger Ramsey claims.

    Args:
        n: Number of vertices.
        p: Observed edge density (p_D).

    Returns:
        Dict with:
          "above_appearance_threshold":       bool  — p > p*_appear
          "above_ramsey_coloring_threshold":  bool  — p > p*_Ramsey
          "p_appearance":                     float — p*_appear value
          "p_ramsey_coloring":                float — p*_Ramsey value
          "note":                             str   — plain-English regime description
    """
    p_appear = triangle_appearance_threshold(n)
    p_ramsey = ramsey_coloring_threshold(n)

    above_appear = p > p_appear
    above_ramsey = p > p_ramsey

    if above_ramsey:
        note = (
            f"p_D={p:.5f} exceeds both the triangle appearance threshold "
            f"(p*~{p_appear:.5f}) and the Rödl-Ruciński Ramsey coloring threshold "
            f"(p*~{p_ramsey:.5f}). The graph is in the regime where G(n,p) almost "
            f"surely forces monochromatic triangles in any 2-coloring. The Z-score "
            f"test additionally shows whether the observed graph has more triangles "
            f"than Erdős-Rényi baseline at this density."
        )
    elif above_appear:
        note = (
            f"p_D={p:.5f} exceeds the triangle appearance threshold "
            f"(p*~{p_appear:.5f}) but is BELOW the Rödl-Ruciński Ramsey coloring "
            f"threshold (p*~{p_ramsey:.5f}). Triangles are expected in G(n,p), but "
            f"the Ramsey coloring property does not necessarily hold. The paper's "
            f"Z-score result is a graph-statistical excess, not a Ramsey coloring claim."
        )
    else:
        note = (
            f"p_D={p:.5f} is BELOW the triangle appearance threshold "
            f"(p*~{p_appear:.5f}). Even in a random G(n,p), few triangles are "
            f"expected. The Z-score result must be interpreted with care: a large "
            f"positive Z may simply reflect that a few observed triangles exceed "
            f"a near-zero expectation."
        )

    return {
        "above_appearance_threshold": bool(above_appear),
        "above_ramsey_coloring_threshold": bool(above_ramsey),
        "p_appearance": float(p_appear),
        "p_ramsey_coloring": float(p_ramsey),
        "note": note,
    }


def sparse_triangle_expected(n: int, p: float) -> float:
    """
    Expected number of triangles in G(n, p).

    E[T] = C(n, 3) * p^3

    Args:
        n: Number of vertices.
        p: Edge probability (edge density of observed graph).

    Returns:
        Expected triangle count (float).
    """
    return comb(n, 3) * (p ** 3)


def sparse_triangle_variance(n: int, p: float) -> float:
    """
    Approximate variance of triangle count in G(n, p).

    Var[T] ≈ C(n,3)*p^3*(1-p^3) + 3*C(n,2)*(n-2)*p^5*(1-p)

    The first term is the variance from independent triples; the second
    corrects for pairs of triangles sharing a common edge (non-independent).

    Args:
        n: Number of vertices.
        p: Edge probability.

    Returns:
        Variance of triangle count (float).
    """
    p3 = p ** 3
    # Variance from independent triples
    var_independent = comb(n, 3) * p3 * (1 - p3)
    # Covariance correction for triangles sharing an edge
    # There are 3*C(n,2)*(n-2) pairs of triangles sharing one edge
    var_covariance = 3 * comb(n, 2) * (n - 2) * (p ** 5) * (1 - p)
    return var_independent + var_covariance


def sparse_triangle_zscore(observed: int, n: int, p: float) -> float:
    """
    Z-score of observed triangle count relative to G(n, p) expectation.

    Z = (observed - E[T]) / sqrt(Var[T])

    Z >> 0: far more triangles than expected in a random graph at this density.
         → Non-random structure in the corpus co-occurrence graph.
    Z ≈ 0: consistent with random graph at the same density.

    Args:
        observed: Observed triangle count.
        n: Number of vertices.
        p: Edge density of the graph.

    Returns:
        Z-score (float). Returns inf if variance is 0.
    """
    expected = sparse_triangle_expected(n, p)
    variance = sparse_triangle_variance(n, p)
    if variance <= 0:
        return float("inf") if observed > expected else 0.0
    return (observed - expected) / sqrt(variance)


def sparse_excess_ratio(observed: int, n: int, p: float) -> float:
    """
    Sparse Excess Ratio (SER): observed / E[T] in G(n, p).

    Analogous to the Ramsey Excess Ratio (RER) but relative to a sparse
    random graph baseline instead of Goodman's complete-graph bound.

    SER > 1: more triangles than expected at this density.
    SER ≈ 1: consistent with random structure at this density.
    SER < 1: fewer triangles than expected (sub-random clustering).

    Args:
        observed: Observed triangle count.
        n: Number of vertices.
        p: Edge density.

    Returns:
        SER value (float). Returns inf if expected is 0.
    """
    expected = sparse_triangle_expected(n, p)
    if expected <= 0:
        return float("inf") if observed > 0 else 1.0
    return observed / expected


def analyze_sparse_bound(
    observed: int,
    n: int,
    p: float,
    tau: float,
    vocab_size: int,
) -> dict:
    """
    Full sparse-corrected triangle analysis for one (tau, vocab_size) data point.

    Args:
        observed: Observed triangle count from Task 2.
        n: Effective vertex count (n_vertices from Task 2).
        p: Edge density (edge_density from Task 2).
        tau: PPMI threshold.
        vocab_size: Vocabulary size for this experiment.

    Returns:
        Dictionary with expected, variance, z_score, sparse_excess_ratio,
        interpretation flags, and threshold_context (from graph_statistical_regime).
    """
    expected = sparse_triangle_expected(n, p)
    variance = sparse_triangle_variance(n, p)
    z = sparse_triangle_zscore(observed, n, p)
    ser = sparse_excess_ratio(observed, n, p)
    threshold_ctx = graph_statistical_regime(n, p)

    return {
        "tau": float(tau),
        "vocab_size": int(vocab_size),
        "n_vertices": int(n),
        "edge_density": float(p),
        "observed_triangles": int(observed),
        "expected_sparse": float(expected),
        "variance_sparse": float(variance),
        "std_sparse": float(variance ** 0.5),
        "z_score": float(z),
        "sparse_excess_ratio": float(ser),
        # Interpretation flags
        "significantly_more_than_random": bool(z > 3.0),   # >3σ above expectation
        "significantly_fewer_than_random": bool(z < -3.0),  # >3σ below expectation
        "consistent_with_random": bool(abs(z) <= 3.0),
        # Honest threshold context — distinguishes appearance, Ramsey coloring, and
        # graph-statistical claims (see module docstring for full explanation)
        "threshold_context": threshold_ctx,
    }
