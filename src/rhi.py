"""
rhi.py — Ramsey Hallucination Index (RHI) computation.

Tests whether hallucinated tokens in GPT-2 outputs on TruthfulQA form
PPMI triangles more often than non-hallucinated tokens. This is the
first empirical test of the Ramsey-hallucination connection.

Methodology:
    1. For each TruthfulQA question, generate GPT-2's answer.
    2. Label the answer as hallucinated / not using TruthfulQA ground truth.
    3. Extract content tokens (remove stopwords).
    4. Check if any triple of content tokens forms a triangle in G_D.
    5. Compute RHI_empirical and run a chi-squared test.

WARNING: The RHI → 1 proposition in 01_framework.pdf is not proven.
We compute RHI_empirical as an *empirical* observation only.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2_contingency
from tqdm import tqdm

# Lazy-loaded sentence-transformer model (cached between calls)
_st_model = None


def _get_st_model():
    """Load and cache the sentence-transformers model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model (all-MiniLM-L6-v2) ...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

# Stopwords — simple English stopword list for content token extraction
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "than", "then", "when",
    "where", "which", "who", "what", "how", "why", "very", "just", "also",
}


def load_truthfulqa(path: str) -> List[Dict]:
    """
    Load TruthfulQA CSV file.

    Expected columns: Question, Best Answer, Correct Answers, Incorrect Answers, etc.

    Args:
        path: Path to TruthfulQA.csv

    Returns:
        List of dicts with keys: question, best_answer, correct_answers, incorrect_answers.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append({
                "question": row.get("Question", "").strip(),
                "best_answer": row.get("Best Answer", "").strip(),
                "correct_answers": row.get("Correct Answers", "").strip(),
                "incorrect_answers": row.get("Incorrect Answers", "").strip(),
                "category": row.get("Category", "").strip(),
            })
    print(f"Loaded {len(records)} TruthfulQA questions from {path}")
    return records


def generate_gpt2_answers(
    questions: List[str],
    max_new_tokens: int = 100,
    batch_size: int = 8,
    device: str = "cpu",
) -> List[str]:
    """
    Generate GPT-2 answers for a list of questions.

    Uses greedy decoding (do_sample=False) for reproducibility.

    Args:
        questions: List of question strings.
        max_new_tokens: Max tokens to generate per answer.
        batch_size: Batch size for generation.
        device: 'cpu' or 'cuda'.

    Returns:
        List of generated answer strings (same length as questions).
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading GPT-2 (device={device}) ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device != "cpu":
        model = model.to(device)

    answers = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating GPT-2 answers"):
        batch_qs = questions[i : i + batch_size]
        inputs = tokenizer(
            batch_qs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            answers.append(text.strip())

    return answers


def is_hallucinated(
    generated_answer: str,
    correct_answers: str,
    incorrect_answers: str,
) -> Optional[bool]:
    """
    Label a generated answer as hallucinated or not.

    Heuristic: an answer is hallucinated if it contains tokens from incorrect_answers
    but not from correct_answers. Returns None if indeterminate.

    Args:
        generated_answer: The text generated by GPT-2.
        correct_answers: Semicolon-separated correct answers from TruthfulQA.
        incorrect_answers: Semicolon-separated incorrect answers.

    Returns:
        True if hallucinated, False if correct, None if indeterminate.
    """
    gen_tokens = set(re.findall(r"\b\w+\b", generated_answer.lower()))

    correct_set: Set[str] = set()
    for ans in correct_answers.split(";"):
        correct_set.update(re.findall(r"\b\w+\b", ans.lower()))

    incorrect_set: Set[str] = set()
    for ans in incorrect_answers.split(";"):
        incorrect_set.update(re.findall(r"\b\w+\b", ans.lower()))

    # Remove stopwords
    correct_set -= _STOPWORDS
    incorrect_set -= _STOPWORDS
    gen_tokens -= _STOPWORDS

    if not gen_tokens:
        return None

    overlap_correct = len(gen_tokens & correct_set)
    overlap_incorrect = len(gen_tokens & incorrect_set)

    if overlap_correct == 0 and overlap_incorrect == 0:
        return None  # Can't determine
    if overlap_incorrect > overlap_correct:
        return True  # Hallucinated
    return False  # Correct


def load_or_build_embeddings(
    records: List[Dict],
    generated_answers: List[str],
    cache_path: str = "data/truthfulqa_embeddings.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached embeddings or compute them with sentence-transformers.

    Embeddings are computed for:
      - generated answers (shape: [N, D])
      - correct answers — one embedding per record (mean of per-sentence embeddings)
      - incorrect answers — one embedding per record (mean of per-sentence embeddings)

    Args:
        records: TruthfulQA records (list of dicts with correct_answers, incorrect_answers).
        generated_answers: GPT-2 generated texts (same length as records).
        cache_path: NPZ path to cache/load embeddings.

    Returns:
        Tuple (gen_embs, correct_embs, incorrect_embs) each of shape [N, D].
    """
    cache = Path(cache_path)
    n = len(records)

    if cache.exists():
        print(f"Loading cached embeddings from {cache} ...")
        data = np.load(str(cache))
        if (
            "gen_embs" in data
            and "correct_embs" in data
            and "incorrect_embs" in data
            and data["gen_embs"].shape[0] == n
        ):
            return data["gen_embs"], data["correct_embs"], data["incorrect_embs"]
        print("Cache size mismatch — recomputing embeddings.")

    model = _get_st_model()

    # Embed generated answers
    print(f"Embedding {n} generated answers ...")
    gen_embs = model.encode(generated_answers, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True)

    # Embed correct and incorrect answer sets
    # Each record may have multiple answers separated by ";"
    correct_embs = np.zeros((n, gen_embs.shape[1]), dtype=np.float32)
    incorrect_embs = np.zeros((n, gen_embs.shape[1]), dtype=np.float32)

    print("Embedding correct/incorrect answer sets ...")
    for i, rec in enumerate(tqdm(records, desc="Embedding answer sets")):
        correct_parts = [s.strip() for s in rec["correct_answers"].split(";") if s.strip()]
        incorrect_parts = [s.strip() for s in rec["incorrect_answers"].split(";") if s.strip()]

        if correct_parts:
            c_embs = model.encode(correct_parts, normalize_embeddings=True)
            correct_embs[i] = c_embs.mean(axis=0)
        if incorrect_parts:
            ic_embs = model.encode(incorrect_parts, normalize_embeddings=True)
            incorrect_embs[i] = ic_embs.mean(axis=0)

    # Save cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache),
        gen_embs=gen_embs,
        correct_embs=correct_embs,
        incorrect_embs=incorrect_embs,
    )
    print(f"Cached embeddings to {cache}")

    return gen_embs, correct_embs, incorrect_embs


def is_hallucinated_semantic(
    gen_emb: np.ndarray,
    correct_embs_row: np.ndarray,
    incorrect_embs_row: np.ndarray,
    threshold: float = 0.1,
) -> Optional[bool]:
    """
    Label a generated answer as hallucinated using semantic cosine similarity.

    The generated answer embedding is compared (dot product, since embeddings
    are L2-normalised) to the mean correct-answer embedding and the mean
    incorrect-answer embedding.

    Label logic:
      - sim_correct  > sim_incorrect + threshold → correct  (False)
      - sim_incorrect > sim_correct  + threshold → hallucinated (True)
      - otherwise                                → None (indeterminate)

    Args:
        gen_emb: L2-normalised embedding of the generated answer, shape [D].
        correct_embs_row: Mean embedding of correct answers, shape [D].
        incorrect_embs_row: Mean embedding of incorrect answers, shape [D].
        threshold: Minimum margin to assign a label (default 0.1).

    Returns:
        True if hallucinated, False if correct, None if indeterminate.
    """
    # Cosine similarity = dot product (embeddings are already normalised)
    sim_correct = float(np.dot(gen_emb, correct_embs_row))
    sim_incorrect = float(np.dot(gen_emb, incorrect_embs_row))

    if sim_correct > sim_incorrect + threshold:
        return False   # Correct
    if sim_incorrect > sim_correct + threshold:
        return True    # Hallucinated
    return None        # Indeterminate


def compute_rhi_semantic(
    records: List[Dict],
    generated_answers: List[str],
    adj: sp.csr_matrix,
    token2id: dict,
    embeddings_cache: str = "data/truthfulqa_embeddings.npz",
    threshold: float = 0.1,
) -> dict:
    """
    Compute RHI using semantic similarity labeling instead of token overlap.

    Replaces the noisy token-overlap ``is_hallucinated`` with a
    sentence-transformer cosine-similarity approach (see ``is_hallucinated_semantic``).

    Args:
        records: TruthfulQA records.
        generated_answers: GPT-2 generated answers.
        adj: Binary sparse adjacency matrix of G_D.
        token2id: Vocabulary mapping.
        embeddings_cache: Path to cache sentence-transformer embeddings.
        threshold: Cosine-similarity margin for labeling (default 0.1).

    Returns:
        Dictionary with RHI, contingency table, chi-squared results, and raw data.
        Each raw_result entry also contains 'sim_correct' and 'sim_incorrect'.
    """
    gen_embs, correct_embs, incorrect_embs = load_or_build_embeddings(
        records, generated_answers, cache_path=embeddings_cache
    )

    results = []

    for i, (rec, answer) in enumerate(tqdm(
        zip(records, generated_answers),
        total=len(records),
        desc="Computing RHI (semantic)",
    )):
        gen_emb = gen_embs[i]
        correct_emb = correct_embs[i]
        incorrect_emb = incorrect_embs[i]

        hallucinated = is_hallucinated_semantic(
            gen_emb, correct_emb, incorrect_emb, threshold=threshold
        )
        if hallucinated is None:
            continue

        token_ids = extract_content_tokens(answer, token2id)
        ids_unique = list(set(token_ids))
        k = len(ids_unique)

        has_triangle = has_ppmi_triangle(ids_unique, adj)
        n_triangles = count_ppmi_triangles(ids_unique, adj)
        max_triangles = k * (k - 1) * (k - 2) // 6
        triangle_density = n_triangles / max(1, max_triangles)

        sim_correct = float(np.dot(gen_emb, correct_emb))
        sim_incorrect = float(np.dot(gen_emb, incorrect_emb))

        results.append({
            "question": rec["question"],
            "generated": answer,
            "category": rec.get("category", ""),
            "hallucinated": hallucinated,
            "has_ppmi_triangle": has_triangle,
            "n_content_tokens": k,
            "n_triangles": n_triangles,
            "triangle_density": float(triangle_density),
            "sim_correct": sim_correct,
            "sim_incorrect": sim_incorrect,
        })

    # Contingency table
    tp = sum(1 for r in results if r["hallucinated"] and r["has_ppmi_triangle"])
    fp = sum(1 for r in results if r["hallucinated"] and not r["has_ppmi_triangle"])
    tn = sum(1 for r in results if not r["hallucinated"] and not r["has_ppmi_triangle"])
    fn = sum(1 for r in results if not r["hallucinated"] and r["has_ppmi_triangle"])

    contingency = np.array([[tp, fp], [fn, tn]])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

    n_hallucinated = tp + fp
    n_correct = fn + tn
    rhi_empirical = tp / n_hallucinated if n_hallucinated > 0 else 0.0
    triangle_rate_correct = fn / n_correct if n_correct > 0 else 0.0

    # Triangle density t-test (continuous RHI 2.0)
    density_stats = _compute_triangle_density_stats(results)

    summary = {
        "labeling_method": "semantic",
        "sim_threshold": threshold,
        "n_labeled": len(results),
        "n_hallucinated": int(n_hallucinated),
        "n_correct": int(n_correct),
        "contingency_table": {
            "hallucinated_with_triangle": int(tp),
            "hallucinated_without_triangle": int(fp),
            "correct_without_triangle": int(tn),
            "correct_with_triangle": int(fn),
        },
        "rhi_empirical": float(rhi_empirical),
        "triangle_rate_correct": float(triangle_rate_correct),
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
        **density_stats,
        "raw_results": results,
    }

    print(f"\nRHI Results (semantic labeling):")
    print(f"  Labeled: {len(results)} answers ({n_hallucinated} hallucinated, {n_correct} correct)")
    print(f"  RHI_empirical: {rhi_empirical:.4f}")
    print(f"  Triangle rate in correct answers: {triangle_rate_correct:.4f}")
    print(f"  Chi-squared: {chi2_stat:.4f}, p-value: {p_value:.6f}")
    print(f"  Significant at p<0.05: {p_value < 0.05}")
    print(f"  --- RHI 2.0 (triangle density) ---")
    print(f"  Mean triangle density (hallucinated): {density_stats['mean_triangle_density_hallucinated']:.6f}")
    print(f"  Mean triangle density (correct):      {density_stats['mean_triangle_density_correct']:.6f}")
    if density_stats["density_t_stat"] is not None:
        print(f"  t-stat: {density_stats['density_t_stat']:.4f}, p-value (one-tailed): {density_stats['density_p_value']:.6f}")
        print(f"  Density significant at p<0.05: {density_stats['density_significant_at_0.05']}")

    return summary


def extract_content_tokens(text: str, token2id: dict) -> List[int]:
    """
    Extract content token IDs from a text string.

    Lowercases, removes stopwords, and keeps only tokens in the vocabulary.

    Args:
        text: Raw text.
        token2id: Vocabulary mapping.

    Returns:
        List of token IDs (may be empty).
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [
        token2id[t]
        for t in tokens
        if t not in _STOPWORDS and t in token2id
    ]


def has_ppmi_triangle(token_ids: List[int], adj: sp.csr_matrix) -> bool:
    """
    Check whether any triple of token IDs forms a triangle in G_D.

    A triangle exists if all three pairwise edges are present in the
    adjacency matrix (i.e., all three PPMI scores > tau).

    Args:
        token_ids: List of token indices to check.
        adj: Binary sparse adjacency matrix of G_D.

    Returns:
        True if at least one triangle exists among the token IDs.
    """
    ids = list(set(token_ids))  # deduplicate
    if len(ids) < 3:
        return False

    adj_csr = adj.tocsr()
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if adj_csr[ids[a], ids[b]] == 0:
                continue
            for c in range(b + 1, len(ids)):
                if adj_csr[ids[a], ids[c]] != 0 and adj_csr[ids[b], ids[c]] != 0:
                    return True
    return False


def count_ppmi_triangles(token_ids: List[int], adj: sp.csr_matrix) -> int:
    """
    Count the number of triangles among a set of token IDs in G_D.

    Uses the same brute-force triple enumeration as has_ppmi_triangle but
    counts all of them. Suitable for the small per-answer token sets
    (typically k < 30) encountered in RHI computation.

    Args:
        token_ids: List of token indices (will be deduplicated).
        adj: Binary sparse adjacency matrix of G_D.

    Returns:
        Number of triangles (each triangle counted once).
    """
    ids = list(set(token_ids))
    if len(ids) < 3:
        return 0

    count = 0
    adj_csr = adj.tocsr()
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if adj_csr[ids[a], ids[b]] == 0:
                continue
            for c in range(b + 1, len(ids)):
                if adj_csr[ids[a], ids[c]] != 0 and adj_csr[ids[b], ids[c]] != 0:
                    count += 1
    return count


def _compute_triangle_density_stats(results: list) -> dict:
    """
    Compute triangle density summary stats and t-test from a raw results list.

    triangle_density = n_triangles / max(1, C(k, 3))
    where k = number of unique in-vocab content tokens in the answer.

    The one-tailed t-test tests H1: mean density(hallucinated) > mean density(correct).

    Args:
        results: List of per-question dicts; each must have 'hallucinated' and
                 'triangle_density' keys.

    Returns:
        Dict with mean densities, t-statistic, and p-value.
    """
    from scipy.stats import ttest_ind

    hall_densities = [r["triangle_density"] for r in results if r["hallucinated"]]
    correct_densities = [r["triangle_density"] for r in results if not r["hallucinated"]]

    mean_hall = float(np.mean(hall_densities)) if hall_densities else 0.0
    mean_correct = float(np.mean(correct_densities)) if correct_densities else 0.0

    density_t_stat = None
    density_p_value = None
    if len(hall_densities) >= 2 and len(correct_densities) >= 2:
        # one-tailed: hallucinated > correct  →  alternative="greater"
        t_result = ttest_ind(hall_densities, correct_densities, alternative="greater")
        density_t_stat = float(t_result.statistic)
        density_p_value = float(t_result.pvalue)

    return {
        "mean_triangle_density_hallucinated": mean_hall,
        "mean_triangle_density_correct": mean_correct,
        "density_t_stat": density_t_stat,
        "density_p_value": density_p_value,
        "density_significant_at_0.05": bool(density_p_value is not None and density_p_value < 0.05),
    }


def compute_rhi(
    records: List[Dict],
    generated_answers: List[str],
    adj: sp.csr_matrix,
    token2id: dict,
) -> dict:
    """
    Compute the empirical Ramsey Hallucination Index.

    For each record, labels the generated answer, extracts content tokens,
    checks for PPMI triangles, and computes triangle_density (RHI 2.0).
    Runs both a chi-squared test (binary) and a t-test (continuous density).

    Args:
        records: TruthfulQA records (from load_truthfulqa).
        generated_answers: GPT-2 generated answers (same length as records).
        adj: Binary sparse adjacency matrix of G_D.
        token2id: Vocabulary mapping.

    Returns:
        Dictionary with RHI, contingency table, chi-squared results,
        triangle density stats, and raw data.
    """
    results = []

    for rec, answer in tqdm(
        zip(records, generated_answers),
        total=len(records),
        desc="Computing RHI",
    ):
        hallucinated = is_hallucinated(
            answer,
            rec["correct_answers"],
            rec["incorrect_answers"],
        )
        if hallucinated is None:
            continue

        token_ids = extract_content_tokens(answer, token2id)
        ids_unique = list(set(token_ids))
        k = len(ids_unique)

        has_triangle = has_ppmi_triangle(ids_unique, adj)
        n_triangles = count_ppmi_triangles(ids_unique, adj)
        max_triangles = k * (k - 1) * (k - 2) // 6
        triangle_density = n_triangles / max(1, max_triangles)

        results.append({
            "question": rec["question"],
            "generated": answer,
            "hallucinated": hallucinated,
            "has_ppmi_triangle": has_triangle,
            "n_content_tokens": k,
            "n_triangles": n_triangles,
            "triangle_density": float(triangle_density),
        })

    # Build contingency table
    # Rows: hallucinated (T/F), Cols: has_triangle (T/F)
    tp = sum(1 for r in results if r["hallucinated"] and r["has_ppmi_triangle"])
    fp = sum(1 for r in results if r["hallucinated"] and not r["has_ppmi_triangle"])
    tn = sum(1 for r in results if not r["hallucinated"] and not r["has_ppmi_triangle"])
    fn = sum(1 for r in results if not r["hallucinated"] and r["has_ppmi_triangle"])

    contingency = np.array([[tp, fp], [fn, tn]])

    # Chi-squared test (binary: has triangle or not)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

    n_hallucinated = tp + fp
    n_correct = fn + tn

    rhi_empirical = tp / n_hallucinated if n_hallucinated > 0 else 0.0
    triangle_rate_correct = fn / n_correct if n_correct > 0 else 0.0

    # Triangle density t-test (continuous RHI 2.0)
    density_stats = _compute_triangle_density_stats(results)

    summary = {
        "n_labeled": len(results),
        "n_hallucinated": int(n_hallucinated),
        "n_correct": int(n_correct),
        "contingency_table": {
            "hallucinated_with_triangle": int(tp),
            "hallucinated_without_triangle": int(fp),
            "correct_without_triangle": int(tn),
            "correct_with_triangle": int(fn),
        },
        "rhi_empirical": float(rhi_empirical),
        "triangle_rate_correct": float(triangle_rate_correct),
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
        **density_stats,
        "raw_results": results,
    }

    print(f"\nRHI Results:")
    print(f"  Labeled: {len(results)} answers ({n_hallucinated} hallucinated, {n_correct} correct)")
    print(f"  RHI_empirical: {rhi_empirical:.4f}")
    print(f"  Triangle rate in hallucinations: {rhi_empirical:.4f}")
    print(f"  Triangle rate in correct answers: {triangle_rate_correct:.4f}")
    print(f"  Chi-squared: {chi2_stat:.4f}, p-value: {p_value:.6f}")
    print(f"  Significant at p<0.05: {p_value < 0.05}")
    print(f"  --- RHI 2.0 (triangle density) ---")
    print(f"  Mean triangle density (hallucinated): {density_stats['mean_triangle_density_hallucinated']:.6f}")
    print(f"  Mean triangle density (correct):      {density_stats['mean_triangle_density_correct']:.6f}")
    if density_stats["density_t_stat"] is not None:
        print(f"  t-stat: {density_stats['density_t_stat']:.4f}, p-value (one-tailed): {density_stats['density_p_value']:.6f}")
        print(f"  Density significant at p<0.05: {density_stats['density_significant_at_0.05']}")

    return summary
