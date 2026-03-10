"""
run_task3_proverbs_analysis.py — Deep-dive into WHY the Proverbs category is significant.

The per-category RHI analysis found that GPT-2 hallucinated answers on TruthfulQA
Proverbs questions contain PPMI triangles significantly more often than correct
answers (p=0.020, RHI=0.857, n=18).

This script traces those triangles back to:
  1. Which specific token triples drive the effect.
  2. How strongly each triple is connected in the PPMI graph (ppmi_min, ppmi_mean).
  3. How often the triple co-occurs in the WikiText-103 training corpus.

The goal is to determine whether the triangles reflect genuine encyclopedic
knowledge clusters (e.g., "apple falls tree") or surface-level clichés
("bird hand bush") that GPT-2 over-generates because they appear densely
in the training data.

Inputs
------
  data/TruthfulQA.csv              — questions + ground truth + category
  data/gpt2_answers.json           — cached GPT-2 generated answers
  data/ppmi_matrix.npz             — PPMI matrix (tau=2.0, vocab=10,000)
  data/wikitext103_train.json      — training corpus (JSONL)

Outputs
-------
  results/task3_proverbs_triangles.json
  results/task3_proverbs_triangles.txt   (human-readable table)

Usage
-----
    python scripts/run_task3_proverbs_analysis.py \\
        --truthfulqa data/TruthfulQA.csv \\
        --answers-cache data/gpt2_answers.json \\
        --ppmi-cache data/ppmi_matrix.npz \\
        --corpus data/wikitext103_train.json \\
        --vocab-size 10000 \\
        --tau 2.0
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import build_vocabulary, iter_sentences
from src.ppmi import load_ppmi, threshold_ppmi
from src.rhi import (
    _STOPWORDS,
    extract_content_tokens,
    is_hallucinated_semantic,
    load_or_build_embeddings,
    load_truthfulqa,
)


# ──────────────────────────────────────────────────────────────────────────────
# Triangle enumeration for a token set
# ──────────────────────────────────────────────────────────────────────────────

def enumerate_ppmi_triangles(
    token_ids: List[int],
    adj,
) -> List[Tuple[int, int, int]]:
    """
    Return all triangles among `token_ids` in the PPMI graph.

    Args:
        token_ids: Deduplicated list of token IDs.
        adj: Binary sparse CSR adjacency matrix.

    Returns:
        List of (id_a, id_b, id_c) tuples, each sorted ascending.
    """
    ids = sorted(set(token_ids))
    if len(ids) < 3:
        return []

    adj_csr = adj.tocsr()
    triangles = []
    for a, b, c in combinations(ids, 3):
        if (adj_csr[a, b] != 0 and adj_csr[a, c] != 0 and adj_csr[b, c] != 0):
            triangles.append((a, b, c))
    return triangles


# ──────────────────────────────────────────────────────────────────────────────
# Corpus co-occurrence lookup
# ──────────────────────────────────────────────────────────────────────────────

def build_triple_cooccurrence(
    corpus_path: str,
    token2id: dict,
    target_triples: Set[Tuple[int, int, int]],
    window: int = 5,
) -> Dict[Tuple[int, int, int], int]:
    """
    Count how often each target triple co-occurs within a sliding window.

    A triple (a, b, c) is counted when all three tokens appear within the
    same `window`-sized window in a corpus sentence. Each window position
    is counted once per unique triple occurrence.

    Args:
        corpus_path: Path to JSONL corpus.
        token2id: Vocabulary mapping.
        target_triples: Set of (id_a, id_b, id_c) tuples to count (sorted).
        window: Half-window size (tokens on each side).

    Returns:
        Dict mapping triple -> count.
    """
    triple_counts: Dict[Tuple[int, int, int], int] = Counter()
    target_set = target_triples  # already a set, O(1) lookup

    for sentence in iter_sentences(corpus_path):
        ids = [token2id[t] for t in sentence if t in token2id]
        if len(ids) < 3:
            continue
        # Slide a window of size 2*window+1 over the sentence
        n = len(ids)
        for center in range(n):
            lo = max(0, center - window)
            hi = min(n, center + window + 1)
            window_ids = set(ids[lo:hi])
            if len(window_ids) < 3:
                continue
            for triple in combinations(sorted(window_ids), 3):
                if triple in target_set:
                    triple_counts[triple] += 1

    return triple_counts


# ──────────────────────────────────────────────────────────────────────────────
# PPMI value lookup
# ──────────────────────────────────────────────────────────────────────────────

def get_pairwise_ppmi(
    ppmi_matrix,
    id_a: int,
    id_b: int,
) -> float:
    """Return the PPMI value for the pair (id_a, id_b)."""
    val = ppmi_matrix[id_a, id_b]
    if hasattr(val, "toarray"):
        val = val.toarray()[0, 0]
    return float(val)


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable table
# ──────────────────────────────────────────────────────────────────────────────

def format_txt_table(data: dict) -> str:
    """Render results/task3_proverbs_triangles.json as a plain-text table."""
    lines = [
        "=" * 78,
        "PROVERBS CATEGORY — TOP HALLUCINATION-SPECIFIC PPMI TRIANGLES",
        "=" * 78,
        f"Proverb questions: {data['n_proverb_questions']}",
        f"Labeled: {data['n_hallucinated']} hallucinated, {data['n_correct']} correct",
        f"RHI (Proverbs): {data['rhi_proverbs']:.3f}",
        "",
        f"{'Rank':<5} {'Tokens':<35} {'Hall':>5} {'Corr':>5} "
        f"{'PPMI_min':>9} {'PPMI_mean':>10} {'Corpus':>8}",
        "-" * 78,
    ]
    for i, t in enumerate(data["top_hallucination_triangles"], 1):
        tokens_str = " — ".join(t["tokens"])
        lines.append(
            f"{i:<5} {tokens_str:<35} {t['frequency_in_hallucinated']:>5} "
            f"{t['frequency_in_correct']:>5} {t['ppmi_min']:>9.3f} "
            f"{t['ppmi_mean']:>10.3f} {t['corpus_cooccurrence_count']:>8}"
        )
    lines += [
        "-" * 78,
        "",
        "Columns:",
        "  Hall   = times this triangle appears in hallucinated answers",
        "  Corr   = times this triangle appears in correct answers",
        "  PPMI_min  = minimum pairwise PPMI among the 3 edges",
        "  PPMI_mean = mean pairwise PPMI",
        "  Corpus = co-occurrence count within 5-token window in WikiText-103",
        "",
        "Interpretation:",
        "  High Corpus count + high Hall + low Corr → cliché phrase GPT-2",
        "  over-generates because it is densely represented in training data.",
        "  This is evidence of the Ramsey mechanism: spurious triangles in the",
        "  PPMI graph drive GPT-2 toward training-data attractors (hallucinations).",
        "=" * 78,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 3 Proverbs: trace PPMI triangles in hallucinated proverb answers"
    )
    parser.add_argument("--truthfulqa", default="data/TruthfulQA.csv")
    parser.add_argument("--answers-cache", default="data/gpt2_answers.json")
    parser.add_argument("--ppmi-cache", default="data/ppmi_matrix.npz")
    parser.add_argument("--corpus", default="data/wikitext103_train.json")
    parser.add_argument("--embeddings-cache", default="data/truthfulqa_embeddings.npz")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--sim-threshold", type=float, default=0.0,
                        help="Cosine-sim margin for semantic labeling (default 0.0 = label all)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top triangles to report (default 20)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading TruthfulQA ...")
    all_records = load_truthfulqa(args.truthfulqa)

    print("Loading GPT-2 answers ...")
    with open(args.answers_cache) as fh:
        all_answers = json.load(fh)
    all_answers = all_answers[:len(all_records)]

    # Filter to Proverbs category
    proverb_pairs = [
        (rec, ans)
        for rec, ans in zip(all_records, all_answers)
        if rec.get("category", "").strip().lower() == "proverbs"
    ]
    if not proverb_pairs:
        print("ERROR: No questions found with Category='Proverbs'. "
              "Check the category field name in TruthfulQA.csv.")
        sys.exit(1)
    proverb_records = [p[0] for p in proverb_pairs]
    proverb_answers = [p[1] for p in proverb_pairs]
    print(f"  {len(proverb_records)} Proverbs questions found.")

    # ── Load PPMI + vocabulary ─────────────────────────────────────────────────
    print(f"Loading PPMI matrix from {args.ppmi_cache} ...")
    ppmi_matrix = load_ppmi(args.ppmi_cache)
    adj = threshold_ppmi(ppmi_matrix, args.tau)

    print(f"Building vocabulary (top {args.vocab_size} tokens) ...")
    token2id, id2token = build_vocabulary(args.corpus, vocab_size=args.vocab_size)

    # ── Semantic labeling ─────────────────────────────────────────────────────
    # Use all records for embeddings (cache covers full dataset)
    print("Loading / computing embeddings ...")
    all_gen_embs, all_correct_embs, all_incorrect_embs = load_or_build_embeddings(
        all_records, all_answers, cache_path=args.embeddings_cache
    )

    # Build index from record to embedding row
    rec_to_idx = {id(rec): i for i, rec in enumerate(all_records)}

    labeled_pairs = []
    for rec, ans in proverb_pairs:
        idx = rec_to_idx[id(rec)]
        label = is_hallucinated_semantic(
            all_gen_embs[idx],
            all_correct_embs[idx],
            all_incorrect_embs[idx],
            threshold=args.sim_threshold,
        )
        if label is None:
            continue
        labeled_pairs.append((rec, ans, label))

    n_hallucinated = sum(1 for _, _, lbl in labeled_pairs if lbl)
    n_correct = sum(1 for _, _, lbl in labeled_pairs if not lbl)
    print(f"  Labeled: {len(labeled_pairs)} ({n_hallucinated} hallucinated, {n_correct} correct)")

    if n_hallucinated == 0:
        print("WARNING: No hallucinated answers found in Proverbs. "
              "Try --sim-threshold 0.0.")

    rhi_proverbs = (
        sum(1 for _, ans, lbl in labeled_pairs
            if lbl and has_any_triangle(ans, token2id, adj))
        / max(1, n_hallucinated)
    )

    # ── Enumerate triangles per answer ────────────────────────────────────────
    hall_triangle_counter: Counter = Counter()
    corr_triangle_counter: Counter = Counter()

    for rec, ans, lbl in labeled_pairs:
        token_ids = list(set(extract_content_tokens(ans, token2id)))
        triangles = enumerate_ppmi_triangles(token_ids, adj)
        counter = hall_triangle_counter if lbl else corr_triangle_counter
        for tri in triangles:
            counter[tri] += 1

    # ── Select top triangles ──────────────────────────────────────────────────
    # Sort by hallucinated frequency, then by (hall - corr) margin
    all_hall_triples = set(hall_triangle_counter.keys())
    top_triples = sorted(
        all_hall_triples,
        key=lambda t: (hall_triangle_counter[t],
                       hall_triangle_counter[t] - corr_triangle_counter.get(t, 0)),
        reverse=True,
    )[:args.top_n]

    print(f"\nTop {len(top_triples)} hallucination triangles identified. "
          "Computing corpus co-occurrence counts (this may take a minute) ...")

    # ── Corpus co-occurrence ─────────────────────────────────────────────────
    target_set = set(top_triples)
    if Path(args.corpus).exists() and target_set:
        corpus_counts = build_triple_cooccurrence(
            args.corpus, token2id, target_set, window=5
        )
    else:
        corpus_counts = {}
        if not Path(args.corpus).exists():
            print(f"WARNING: Corpus not found at {args.corpus}. "
                  "Corpus co-occurrence counts will be 0.")

    # ── Build output records ──────────────────────────────────────────────────
    top_triangle_records = []
    for tri in top_triples:
        id_a, id_b, id_c = tri
        tok_a = id2token.get(id_a, f"<{id_a}>")
        tok_b = id2token.get(id_b, f"<{id_b}>")
        tok_c = id2token.get(id_c, f"<{id_c}>")

        ppmi_ab = get_pairwise_ppmi(ppmi_matrix, id_a, id_b)
        ppmi_ac = get_pairwise_ppmi(ppmi_matrix, id_a, id_c)
        ppmi_bc = get_pairwise_ppmi(ppmi_matrix, id_b, id_c)

        top_triangle_records.append({
            "tokens": [tok_a, tok_b, tok_c],
            "token_ids": list(tri),
            "frequency_in_hallucinated": int(hall_triangle_counter[tri]),
            "frequency_in_correct": int(corr_triangle_counter.get(tri, 0)),
            "ppmi_12": float(ppmi_ab),
            "ppmi_13": float(ppmi_ac),
            "ppmi_23": float(ppmi_bc),
            "ppmi_min": float(min(ppmi_ab, ppmi_ac, ppmi_bc)),
            "ppmi_mean": float((ppmi_ab + ppmi_ac + ppmi_bc) / 3),
            "corpus_cooccurrence_count": int(corpus_counts.get(tri, 0)),
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "tau": args.tau,
        "vocab_size": args.vocab_size,
        "sim_threshold": args.sim_threshold,
        "n_proverb_questions": len(proverb_records),
        "n_labeled": len(labeled_pairs),
        "n_hallucinated": int(n_hallucinated),
        "n_correct": int(n_correct),
        "rhi_proverbs": float(rhi_proverbs),
        "n_unique_hall_triangles": len(all_hall_triples),
        "top_hallucination_triangles": top_triangle_records,
    }

    json_path = results_dir / "task3_proverbs_triangles.json"
    with json_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Saved JSON to {json_path}")

    # ── Save TXT ──────────────────────────────────────────────────────────────
    txt_path = results_dir / "task3_proverbs_triangles.txt"
    txt_path.write_text(format_txt_table(output))
    print(f"Saved text table to {txt_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + format_txt_table(output))


# ── Helper used in main ───────────────────────────────────────────────────────

def has_any_triangle(answer: str, token2id: dict, adj) -> bool:
    """Check if an answer text has any PPMI triangle (used for RHI computation)."""
    ids = list(set(extract_content_tokens(answer, token2id)))
    return bool(enumerate_ppmi_triangles(ids, adj))


if __name__ == "__main__":
    main()
