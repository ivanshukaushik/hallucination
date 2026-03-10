"""
ppmi.py — PPMI matrix construction and PPMI graph (G_D).

Computes Positive Pointwise Mutual Information (PPMI) between token pairs
from a co-occurrence matrix, then thresholds to create the correlation graph G_D.

PPMI(t_i, t_j) = max(log2(P(t_i, t_j) / (P(t_i) * P(t_j))), 0)

Reference:
    Church & Hanks (1990). Word Association Norms, Mutual Information, and Lexicography.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.corpus import build_vocabulary, iter_windows


def build_cooccurrence_matrix(
    corpus_path: str,
    token2id: dict,
    window: int = 5,
    text_field: str = "text",
) -> sp.csr_matrix:
    """
    Build a sparse token co-occurrence count matrix.

    Args:
        corpus_path: Path to JSONL corpus.
        token2id: Vocabulary mapping token -> index.
        window: Half-window size for co-occurrence.
        text_field: JSON field name.

    Returns:
        Symmetric (V x V) sparse CSR matrix of co-occurrence counts.
    """
    V = len(token2id)
    print(f"Building co-occurrence matrix (vocab={V}, window={window}) ...")

    rows, cols, data = [], [], []
    # Use a dict to accumulate counts before converting to sparse
    counts: Dict[Tuple[int, int], int] = {}

    for ci, cj in tqdm(iter_windows(corpus_path, token2id, window, text_field), desc="Co-occurrences"):
        key = (min(ci, cj), max(ci, cj))
        counts[key] = counts.get(key, 0) + 1

    for (i, j), cnt in counts.items():
        rows.append(i)
        cols.append(j)
        data.append(cnt)
        if i != j:
            rows.append(j)
            cols.append(i)
            data.append(cnt)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(V, V), dtype=np.float64)
    print(f"Co-occurrence matrix: {matrix.nnz} non-zeros out of {V**2} possible pairs")
    return matrix


def compute_ppmi_matrix(cooc: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute the PPMI matrix from a co-occurrence count matrix.

    PPMI(i, j) = max(log2(P(i,j) / (P(i) * P(j))), 0)

    Args:
        cooc: Symmetric sparse co-occurrence count matrix (V x V).

    Returns:
        Sparse PPMI matrix (same shape). Zero entries mean PPMI = 0.
    """
    total = cooc.sum()
    if total == 0:
        raise ValueError("Co-occurrence matrix is all zeros.")

    # Marginal probabilities P(t_i) = row_sum / total
    row_sums = np.asarray(cooc.sum(axis=1)).flatten()
    p_i = row_sums / total  # shape (V,)

    # Convert to LIL for efficient element-wise operation, then back to CSR
    ppmi = cooc.astype(np.float64).tocsr(copy=True)
    ppmi /= total  # now ppmi[i,j] = P(i,j)

    # Divide each element by P(i) * P(j)
    # Equivalent to: ppmi[i,j] /= p_i[i] * p_i[j]
    # Use D^{-1} A D^{-1} where D = diag(p_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_p = np.where(p_i > 0, 1.0 / p_i, 0.0)

    D_inv = sp.diags(inv_p)
    ppmi = D_inv @ ppmi @ D_inv

    # Take log2, clamp negatives to 0
    ppmi = ppmi.tocsr()
    ppmi.data = np.log2(ppmi.data, where=ppmi.data > 0, out=np.zeros_like(ppmi.data))
    ppmi.data = np.maximum(ppmi.data, 0.0)
    ppmi.eliminate_zeros()

    print(f"PPMI matrix: {ppmi.nnz} non-zero entries")
    return ppmi


def threshold_ppmi(ppmi: sp.csr_matrix, tau: float) -> sp.csr_matrix:
    """
    Threshold the PPMI matrix to get the adjacency matrix of G_D.

    An edge (i, j) exists iff PPMI(i, j) > tau.

    Args:
        ppmi: PPMI sparse matrix.
        tau: Threshold value (e.g. 1.0, 2.0, 3.0).

    Returns:
        Binary sparse adjacency matrix (0/1 floats).
    """
    adj = ppmi.copy()
    adj.data = (adj.data > tau).astype(np.float64)
    adj.eliminate_zeros()
    # Ensure symmetry (PPMI should be symmetric, but enforce it)
    adj = (adj + adj.T)
    adj.data = np.minimum(adj.data, 1.0)
    adj.eliminate_zeros()
    return adj.tocsr()


def graph_stats(adj: sp.csr_matrix) -> dict:
    """
    Compute basic graph statistics for the PPMI graph.

    Args:
        adj: Binary sparse adjacency matrix.

    Returns:
        Dictionary with n_vertices, n_edges, edge_density, degree statistics.
    """
    n = adj.shape[0]
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    n_edges = int(adj.nnz) // 2  # undirected

    max_edges = n * (n - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    return {
        "n_vertices": int(n),
        "n_edges": int(n_edges),
        "edge_density": float(density),
        "degree_mean": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "degree_max": int(degrees.max()),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
    }


def save_ppmi(ppmi: sp.csr_matrix, path: str) -> None:
    """Save PPMI matrix in scipy sparse npz format."""
    sp.save_npz(path, ppmi)
    print(f"Saved PPMI matrix to {path}")


def load_ppmi(path: str) -> sp.csr_matrix:
    """Load PPMI matrix from npz file."""
    return sp.load_npz(path)
