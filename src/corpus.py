"""
corpus.py — Corpus loading and tokenization.

Handles WikiText-103 loading and simple whitespace+punctuation tokenization.
No BPE — we want interpretable token units for the PPMI graph.
"""

import re
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

from tqdm import tqdm


# Punctuation characters to split on (keep as separate tokens)
_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    """
    Tokenize a string with whitespace + punctuation splitting.

    Lowercases, splits on whitespace, and treats each punctuation character
    as a separate token. Empty strings are dropped.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase token strings.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    tokens = _WHITESPACE_RE.split(text.strip())
    return [t for t in tokens if t]


def iter_sentences(corpus_path: str, text_field: str = "text") -> Iterator[List[str]]:
    """
    Yield tokenized sentences from a JSONL corpus file.

    Each line of the file should be a JSON object with a text field.
    Skips empty lines and Wikipedia section headers (lines starting with '=').

    Args:
        corpus_path: Path to JSONL file (e.g., wikitext103_train.json).
        text_field: Key in the JSON object containing the raw text.

    Yields:
        Lists of tokens, one per non-empty text line.
    """
    path = Path(corpus_path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_field, "")
            except json.JSONDecodeError:
                text = line

            if not text.strip() or text.strip().startswith("="):
                continue

            tokens = tokenize(text)
            if tokens:
                yield tokens


def build_vocabulary(
    corpus_path: str,
    vocab_size: int = 10_000,
    min_freq: int = 5,
    text_field: str = "text",
) -> Tuple[dict, dict]:
    """
    Build a vocabulary from the corpus, keeping the top `vocab_size` tokens.

    Args:
        corpus_path: Path to JSONL corpus.
        vocab_size: Maximum vocabulary size (top-N by frequency).
        min_freq: Minimum frequency to include a token.
        text_field: JSON field containing text.

    Returns:
        Tuple of:
            token2id: dict mapping token string -> integer index
            id2token: dict mapping integer index -> token string
    """
    print(f"Building vocabulary from {corpus_path} ...")
    freq: Counter = Counter()

    for sentence in tqdm(iter_sentences(corpus_path, text_field), desc="Counting tokens"):
        freq.update(sentence)

    # Filter by minimum frequency then take top-N
    filtered = [(tok, cnt) for tok, cnt in freq.items() if cnt >= min_freq]
    filtered.sort(key=lambda x: -x[1])
    selected = filtered[:vocab_size]

    token2id = {tok: idx for idx, (tok, _) in enumerate(selected)}
    id2token = {idx: tok for tok, idx in token2id.items()}

    print(f"Vocabulary: {len(token2id)} tokens (min_freq={min_freq}, vocab_size={vocab_size})")
    return token2id, id2token


def iter_windows(
    corpus_path: str,
    token2id: dict,
    window: int = 5,
    text_field: str = "text",
) -> Iterator[Tuple[int, int]]:
    """
    Yield (center_id, context_id) pairs for all co-occurrence windows.

    Uses a symmetric window: for each token at position i, pairs it with
    tokens at positions i-window to i+window (excluding i itself), where
    both tokens are in the vocabulary.

    Args:
        corpus_path: Path to JSONL corpus.
        token2id: Vocabulary mapping.
        window: Half-window size (number of tokens on each side).
        text_field: JSON field name.

    Yields:
        Tuples (center_id, context_id).
    """
    for sentence in iter_sentences(corpus_path, text_field):
        ids = [token2id[t] for t in sentence if t in token2id]
        n = len(ids)
        for i in range(n):
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i != j:
                    yield ids[i], ids[j]
