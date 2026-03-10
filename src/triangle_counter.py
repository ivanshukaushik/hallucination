"""
triangle_counter.py — Triangle counting and comparison to the Goodman lower bound.

The Goodman bound gives the minimum number of monochromatic triangles in any
2-coloring of the complete graph K_n. We compare observed PPMI triangles
against this bound to compute the Ramsey Excess Ratio (RER).

Exact Goodman formula (see Goodman 1959, corrected from approximate version):
    T(n) = C(n,3) - floor(n/2) * floor((n-1)^2 / 4)

Note: The framework paper (01_framework.pdf) erroneously uses n(n-1)(n-5)/24.
This module uses the exact formula.

References:
    Goodman (1959). On Sets of Acquaintances and Strangers. Amer. Math. Monthly 66.
    Pawliuk & Waddell (2019). arXiv:1712.09471.
"""

from math import comb
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def goodman_lower_bound(n: int) -> int:
    """
    Exact Goodman lower bound on monochromatic triangles in any 2-coloring of K_n.

    T(n) = C(n,3) - floor(n/2) * floor((n-1)^2 / 4)

    This is the minimum number of monochromatic triangles guaranteed to exist
    regardless of how the complete graph K_n is 2-colored (edge-colored).

    Args:
        n: Number of vertices.

    Returns:
        Minimum guaranteed number of monochromatic triangles (integer >= 0).
    """
    if n < 3:
        return 0
    total_triples = comb(n, 3)
    subtracted = (n // 2) * ((n - 1) ** 2 // 4)
    return max(0, total_triples - subtracted)


def count_triangles_matrix(adj: sp.csr_matrix) -> int:
    """
    Count the number of triangles in an undirected graph via matrix multiplication.

    The number of triangles is trace(A^3) / 6, where A is the adjacency matrix.
    This is efficient for sparse matrices.

    Args:
        adj: Binary sparse adjacency matrix (symmetric).

    Returns:
        Total number of triangles (each counted once).
    """
    # A^2
    a2 = adj @ adj
    # trace(A^3) = sum of element-wise product of A^2 and A
    # = sum_ij A^2[i,j] * A[j,i]
    # For symmetric A this is sum_ij A^2[i,j] * A[i,j]
    trace_a3 = (a2.multiply(adj)).sum()
    return int(trace_a3) // 6


def count_triangles_per_vertex(adj: sp.csr_matrix) -> np.ndarray:
    """
    Count triangles involving each vertex.

    Uses the diagonal of A^3 / 2 (each triangle appears twice per vertex).

    Args:
        adj: Binary sparse adjacency matrix.

    Returns:
        Array of shape (n,) with triangle count per vertex.
    """
    a2 = adj @ adj
    a3_diag = np.asarray((a2.multiply(adj)).sum(axis=1)).flatten()
    return a3_diag // 2


def ramsey_excess_ratio(observed: int, n: int) -> float:
    """
    Compute the Ramsey Excess Ratio (RER).

    RER = observed_triangles / goodman_lower_bound(n)

    RER >> 1: corpus has far more triangles than the minimum guarantee (expected).
    RER ≈ 1: corpus is near the combinatorial floor (theoretically interesting).

    Args:
        observed: Observed triangle count in G_D.
        n: Number of vertices (vocabulary size at this threshold).

    Returns:
        RER value (float). Returns inf if Goodman bound is 0.
    """
    bound = goodman_lower_bound(n)
    if bound == 0:
        return float("inf") if observed > 0 else 1.0
    return observed / bound


def find_phase_transition(
    ns: List[int],
    edge_densities: List[float],
    c: float = 1.0,
) -> Optional[int]:
    """
    Find the empirical phase transition threshold n* from Rödl-Ruciński (1993).

    The Rödl-Ruciński threshold for triangle appearance in G(n, p) is p >> c/n.
    n* = smallest n such that p_D(n) >> c/n, i.e., p_D(n) * n > c.

    Args:
        ns: List of vocabulary sizes.
        edge_densities: Corresponding edge densities.
        c: Threshold constant (default 1.0).

    Returns:
        First n where p_D * n > c, or None if no such n found.
    """
    for n, p in zip(ns, edge_densities):
        if p * n > c:
            return n
    return None


def analyze_triangles(
    adj: sp.csr_matrix,
    tau: float,
    verbose: bool = True,
) -> dict:
    """
    Full triangle analysis for a given PPMI threshold.

    Counts observed triangles, computes Goodman bound, RER, and phase transition
    info for the adjacency graph at threshold tau.

    Args:
        adj: Binary sparse adjacency matrix of G_D at threshold tau.
        tau: The PPMI threshold used to create adj.
        verbose: Print progress.

    Returns:
        Dictionary of analysis results.
    """
    n = adj.shape[0]
    n_edges = int(adj.nnz) // 2
    max_edges = n * (n - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    if verbose:
        print(f"tau={tau}: n={n}, edges={n_edges}, density={density:.6f}")
        print("Counting triangles (matrix method)...")

    observed = count_triangles_matrix(adj)
    bound = goodman_lower_bound(n)
    rer = ramsey_excess_ratio(observed, n)

    # Rödl-Ruciński check: p_D * n vs 1
    rodl_rucinski_product = density * n

    results = {
        "tau": tau,
        "n_vertices": int(n),
        "n_edges": int(n_edges),
        "edge_density": float(density),
        "observed_triangles": int(observed),
        "goodman_lower_bound": int(bound),
        "ramsey_excess_ratio": float(rer),
        "rodl_rucinski_product": float(rodl_rucinski_product),
        "above_rodl_rucinski_threshold": bool(rodl_rucinski_product > 1.0),
    }

    if verbose:
        print(f"  Observed triangles: {observed:,}")
        print(f"  Goodman lower bound: {bound:,}")
        print(f"  Ramsey Excess Ratio: {rer:.4f}")
        print(f"  Rödl-Ruciński product p_D*n: {rodl_rucinski_product:.4f} ({'above' if rodl_rucinski_product > 1.0 else 'below'} threshold)")

    return results
