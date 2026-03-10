# Ramsey-Theoretic Bounds on LLM Hallucinations

**Status: Active Research (March 2026)**

A formal investigation into whether a provable lower bound on LLM hallucinations exists — a theorem, not a bug.

---

## Core Hypothesis

A subset of LLM hallucinations is structurally inevitable, arising from combinatorial properties of the training corpus guaranteed by Ramsey Theory. No amount of RLHF, fine-tuning, or data augmentation can eliminate them.

The mathematical chain:

1. **Ramsey's Theorem (1930):** In any sufficiently large system, complete disorder is impossible — certain patterns must appear.
2. **Goodman's Theorem (1959):** In any 2-coloring of K_n, the number of monochromatic triangles is at least:

   ```
   T(n) = C(n,3) - floor(n/2) * floor((n-1)^2 / 4)
   ```

3. **Calude & Longo (2016):** Any dataset large enough must contain spurious correlations of any type — guaranteed by size alone.

**Our contribution:** Token co-occurrence graphs in large corpora must contain a minimum number of *spurious triangles* — token triples that are highly co-associated but lack causal grounding. We claim these are the structural origin of a provable class of hallucinations.

---

## Hallucination Taxonomy

| Type | Origin | Mitigable? |
|------|--------|------------|
| **I — Ramsey-Inevitable** | Corpus size (Goodman bound) | **No** — theorem-limited floor |
| **II — Distribution-Induced** | Non-uniform training distribution | Partially — data curation |
| **III — Decoding-Stochastic** | Sampling randomness at inference | Yes — RLHF, greedy decoding |

---

## Key Definitions

**Correlation Graph G_D:** Vertices = token types in vocabulary V. Edge (t_i, t_j) exists iff PPMI(t_i, t_j) > τ.

**Ramsey Hallucination Index (RHI):** RHI(M, D) = |H_I(M,D)| / |H(M,D)| — the fraction of hallucinations that are theorem-limited.

---

## Repository Structure

```
hallucination/
├── README.md
├── requirements.txt
├── .gitignore
├── papers/
│   ├── 01_framework.pdf          # Initial mathematical framework
│   └── 02_critique_and_roadmap.pdf  # Mathematical audit + research plan
├── src/
│   ├── __init__.py
│   ├── corpus.py                 # Corpus loading + tokenization
│   ├── ppmi.py                   # PPMI matrix construction
│   ├── triangle_counter.py       # Triangle counting + Goodman bound
│   └── rhi.py                    # RHI computation
├── scripts/
│   ├── run_task1_ppmi.py         # Task 1: Build PPMI graph on WikiText-103
│   ├── run_task2_triangles.py    # Task 2: Count triangles vs Goodman bound
│   └── run_task3_rhi.py          # Task 3: Test against TruthfulQA
├── results/                      # Output data and plots
└── data/                         # Raw data (gitignored)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download WikiText-103
python -c "from datasets import load_dataset; d = load_dataset('wikitext', 'wikitext-103-raw-v1'); d['train'].to_json('data/wikitext103_train.json')"

# Download TruthfulQA
wget https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv -O data/TruthfulQA.csv

# Task 1: Build PPMI graph and characterize structure
python scripts/run_task1_ppmi.py --corpus data/wikitext103_train.json --vocab-size 10000 --window 5 --tau 2.0

# Task 2: Count triangles vs Goodman bound
python scripts/run_task2_triangles.py --tau 1.0 2.0 3.0

# Task 3: Empirically test RHI on TruthfulQA
python scripts/run_task3_rhi.py --truthfulqa data/TruthfulQA.csv
```

---

## Mathematical Caveats

The framework contains four known gaps (documented in `papers/02_critique_and_roadmap.pdf`):

1. **Goodman formula:** Use exact form `C(n,3) - floor(n/2)*floor((n-1)^2/4)`, not the approximate `n(n-1)(n-5)/24`.
2. **Sparse graphs:** Token graphs are sparse, not K_n. The correct framework uses Rödl-Ruciński (1993) sparse Ramsey bounds.
3. **RHI → 1 claim:** The strong proposition is unprovable as stated. The weaker `S(D)/C(|V|,3) → 1` is provable.
4. **Phase transition threshold:** Anchor to the Rödl-Ruciński edge-density threshold `p_D >> c*n^(-1)`.

Nothing in the theoretical framework should be presented as a proven theorem until formally verified.

---

## Key References

- Calude & Longo (2017). *The Deluge of Spurious Correlations in Big Data.* Foundations of Science.
- Goodman (1959). *On Sets of Acquaintances and Strangers.* Amer. Math. Monthly 66.
- Pawliuk & Waddell (2019). *Using Ramsey Theory to Measure Unavoidable Spurious Correlations in Big Data.* Axioms 8(1). arXiv:1712.09471
- Rödl & Ruciński (1993). *Threshold Functions for Ramsey Properties.* JAMS.
- Coveney & Succi (2025). *The Wall Confronting Large Language Models.* arXiv:2507.19703.
