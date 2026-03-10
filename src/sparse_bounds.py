"""
sparse_bounds.py — Sparse-corrected triangle bounds for G(n, p) random graphs.

The Goodman bound applies to complete graphs K_n. For the actual sparse
co-occurrence graph G(n, p_D) produced by PPMI thresholding, the correct
theoretical baseline is the expected triangle count in a random graph with
the same edge density p_D.

Expected triangles in G(n, p):
    E[T] = C(n, 3) * p^3

Variance (using indicator variables for each triple):
    Var[T] ≈ C(n,3)*p^3*(1-p^3) + 3*C(n,2)*(n-2)*p^5*(1-p)

The second term accounts for the covariance between triangles sharing an edge.

References:
    Bollobás, B. (2001). Random Graphs. Cambridge University Press.
    Rödl & Ruciński (1993). Lower bounds on probability thresholds for Ramsey
        properties. Contemporary Mathematics 144, 317–346.
"""

from math import comb, sqrt
from typing import Tuple


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
        and interpretation flags.
    """
    expected = sparse_triangle_expected(n, p)
    variance = sparse_triangle_variance(n, p)
    z = sparse_triangle_zscore(observed, n, p)
    ser = sparse_excess_ratio(observed, n, p)

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
    }
