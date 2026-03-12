# Claude Code Prompt — Theory Implementation + Paper Rewrite

## Context

This is a research project on mathematical lower bounds for LLM hallucinations.
The codebase is at the root of this repo. We have just adopted a new, disciplined
theoretical framework (information-theoretic, not purely Ramsey graph-statistical).

The user pasted a LaTeX theory draft whose core claims are:

**Theorem 1 (Irreducible hallucination floor):**
  For a text-only learner, if ||P_0^n − P_1^n||_TV ≤ δ_n, then
    P(f(X_n) ≠ Y) ≥ (1 − δ_n) / 2

**Corollary (Rare-evidence bound):**
  R*_n ≥ (1/2) · (1 − ρ)^n
  where ρ = P(a single token contains world-revealing evidence for Y)

**Ambiguous-region bound:**
  R*_n ≥ ε_n · P(X_n ∈ A_n)
  where A_n is the set of observable states that are truth-ambiguous,
  ε_n = E[P(Y=1|X_n) · P(Y=0|X_n) | X_n ∈ A_n] is ambiguity severity.

**What PPMI triangles are (repositioned):**
  Not the theorem — they are *mechanism witnesses*. A PPMI triangle
  (t_i, t_j, t_k) with high PPMI but zero/low local co-occurrence is a
  concrete example of a semantic collapse that contributes to the hallucination
  floor. But the floor itself is proven information-theoretically.

**Safe claims (what the paper CAN say):**
  1. An irreducible hallucination floor exists for any text-only learner
     unable to distinguish P_0 from P_1 from its training corpus.
  2. PPMI motifs may serve as corpus-observable witnesses of the ambiguous region.
  3. n*=10 is the corpus-specific threshold for non-random PPMI structure in
     WikiText-103; it is NOT a universal Ramsey number.

**Unsafe claims (what the paper CANNOT say):**
  1. All hallucinations are unavoidable (only the lower-bounded fraction).
  2. PPMI triangle presence ⟹ hallucination (motifs are correlated, not causal).
  3. n*=10 is a universal constant.

---

## Task 1 — Implement `src/theory.py`

Create a new file `src/theory.py` that implements the formal quantities from
the theory draft. This module should be clean, well-documented, and importable
by the estimation scripts.

```python
# src/theory.py
"""
theory.py — Information-theoretic hallucination floor bounds.

Implements the formal quantities from:
  "A Disciplined Theory Draft for Irreducible Hallucination in
   Text-Only Language Modeling"

All functions are pure-Python with no external dependencies beyond math/numpy.

THEOREM 1 (Irreducible floor):
  For a text-only binary learner f: X^n → {0,1},
  if ||P_0^n − P_1^n||_TV ≤ δ_n, then
      R*_n := min_f P(f(X_n) ≠ Y) ≥ (1 − δ_n) / 2

RARE-EVIDENCE COROLLARY:
  If each token in X_n independently reveals the true world with probability ρ,
  then   δ_n ≤ 2(1 − (1 − ρ)^n)   so   R*_n ≥ (1/2)(1 − ρ)^n.

AMBIGUOUS-REGION BOUND:
  R*_n ≥ ε_n · α_n
  where:
    α_n = P(X_n ∈ A_n)            (mass of ambiguous-context region)
    ε_n = E[p(1−p) | X_n ∈ A_n]  (average ambiguity severity; p = P(Y=1|X_n))
    ε_n · α_n is a computable lower bound on Bayes risk.
"""
```

Implement the following functions:

### `hallucination_floor(delta_n: float) -> float`
  Returns (1 − delta_n) / 2.
  The information-theoretic lower bound on P(f(X_n) ≠ Y) given that the
  TV distance between the two world distributions is ≤ delta_n.
  Raises ValueError if delta_n < 0 or delta_n > 1.

### `tv_distance_upper_bound(rho: float, n: int) -> float`
  Returns 2 * (1 − (1 − rho)^n).
  Upper bound on ||P_0^n − P_1^n||_TV under the rare-evidence model:
  each of the n tokens independently reveals the true world with probability ρ.
  When this is plugged into hallucination_floor, you get the rare-evidence bound.
  Raises ValueError if rho < 0 or rho > 1 or n < 1.

### `rare_evidence_bound(rho: float, n: int) -> float`
  Returns (1/2) * (1 − rho)^n.
  The Rare-Evidence Corollary bound on R*_n directly.
  This is hallucination_floor(tv_distance_upper_bound(rho, n)) simplified.
  Raises ValueError if rho < 0 or rho > 1 or n < 1.

### `ambiguous_region_bound(alpha_n: float, epsilon_n: float) -> float`
  Returns epsilon_n * alpha_n.
  Lower bound on Bayes risk from the ambiguous region.
  alpha_n: fraction of observable contexts that are truth-ambiguous (0 to 1).
  epsilon_n: average per-context ambiguity severity (0 to 0.25 — max at p=0.5).
  Raises ValueError if either argument is out of [0,1] or epsilon_n > 0.25.

### `rho_from_category_frequencies(category_question_pairs: list[tuple[str, float]]) -> float`
  Estimates ρ from per-category rare-evidence rates.
  Each tuple is (category_name, empirical_ρ_for_category).
  Returns the weighted average ρ across categories.
  empirical ρ for a category ≈ fraction of questions in that category for which
  a single token in the context is sufficient to determine the answer.
  (In practice: use the fraction of questions where the correct answer has
  sim_correct ≥ 0.5 in the first token of the question.)

### `compute_delta_n_from_data(n_contexts: int, n_ambiguous: int, epsilon_values: list[float]) -> dict`
  Estimates δ_n from data.
  n_contexts: total number of (question, answer) contexts evaluated
  n_ambiguous: number where sim_correct ≈ sim_incorrect (|sim_c − sim_i| < 0.1)
  epsilon_values: list of p*(1−p) values for each ambiguous context,
                  where p = sim_correct / (sim_correct + sim_incorrect)
  Returns dict with:
    alpha_n: n_ambiguous / n_contexts
    epsilon_n: mean of epsilon_values (or 0 if empty)
    ambiguous_region_bound: epsilon_n * alpha_n
    delta_n_estimate: 1 − 2 * epsilon_n * alpha_n (implied TV distance)
    hallucination_floor: (1 − delta_n_estimate) / 2

### `floor_sensitivity_table(rho_values: list[float], n_values: list[int]) -> list[dict]`
  Returns a table of rare_evidence_bound values for all combinations of
  rho in rho_values and n in n_values.
  Each row: {"rho": float, "n": int, "floor": float, "floor_pct": str}.

---

## Task 2 — Create `scripts/run_theory_estimation.py`

This script loads the existing results and estimates the empirical theory
quantities (α_n, ε_n, ρ) from them. It produces:
- `results/theory_estimation.json` — all computed quantities
- `results/theory_floor_table.json` — sensitivity table for different ρ and n values
- Console output summarising the bounds

```
python scripts/run_theory_estimation.py \
  --semantic-results results/task3_rhi_gpt2_semantic.json \
  --category-results results/task3_rhi_by_category.json \
  --output-dir results/
```

The script should:

1. **Load semantic labeling results** from task3_rhi_gpt2_semantic.json.
   - For each labeled answer, compute p = sim_correct / (sim_correct + sim_incorrect)
   - If that key isn't in the JSON, approximate: p = 0.5 + 0.5 * (is_correct − 0.5)
   - Classify as "ambiguous" if |sim_correct − sim_incorrect| < 0.1 (or if not
     available, if sim_correct is between 0.3 and 0.7)
   - Compute epsilon_values = [p*(1−p) for each ambiguous context]
   - Call compute_delta_n_from_data() to get α_n, ε_n, floor

2. **Estimate ρ per category** from task3_rhi_by_category.json.
   - For each category, ρ_cat ≈ fraction of questions where correct answer
     appears to be "strongly signaled" (use RHI < 0.3 as proxy — categories
     where few hallucinated answers have triangles suggest the tokens clearly
     signal the answer)
   - Report a table of per-category ρ estimates

3. **Print a summary** with:
   - Estimated α_n, ε_n, ambiguous_region_bound
   - Rare-evidence bound for n=1,5,10,50,100,500 tokens at estimated ρ
   - "The hallucination floor from Theorem 1 is at least X%"

4. **Write results/theory_estimation.json** with all quantities.

5. **Write results/theory_floor_table.json** with the sensitivity table for
   ρ ∈ {0.001, 0.005, 0.01, 0.05, 0.1} and n ∈ {10, 50, 100, 200, 500, 1000}.

---

## Task 3 — Rewrite `results/hallucination_paper.docx`

**IMPORTANT:** The paper must be completely rewritten to reflect the new framework.
The old paper led with Ramsey theory empirics; the new paper leads with the
information-theoretic theorem.

The paper should have exactly these sections (use this structure):

### 1. Abstract (200 words)
We prove a lower bound on the irreducible hallucination rate of any text-only
language model. Under mild assumptions (binary classification, TV distance
between latent world distributions bounded by δ_n), the Bayes risk floor is
≥ (1 − δ_n)/2. Under a rare-evidence model, this implies a floor of
(1/2)(1−ρ)^n. We then describe a corpus-structural mechanism — PPMI triangle
motifs — that may serve as observable witnesses of contexts in the ambiguous
region, and present empirical evidence that this mechanism is operative in
proverb-domain hallucinations from GPT-2 on TruthfulQA (n=18, p=0.044). We
measure n*=10 as the corpus-specific phase transition for non-random PPMI
structure in WikiText-103.

### 2. Introduction (600 words)
- LLMs hallucinate; prior explanations are post-hoc
- Our contribution: a *provable lower bound*, not an empirical correlation
- Two-part structure: the theorem (Sec 3) and the mechanism (Sec 4)
- Explicitly state what we can and cannot claim (safe/unsafe claims table)

### 3. Formal Framework (800 words)
- Setup: latent world W ∈ {0,1}, text X_n, text-only learner f
- TV distance definition and why it bounds Bayes risk (Theorem 1 with proof sketch)
- Rare-evidence corollary (with proof)
- Ambiguous-region bound (with proof sketch)
- Include the safe/unsafe claims table from the LaTeX draft

### 4. Corpus Structure as Mechanism (600 words)
- Repositioned: PPMI triangles are mechanism, not theorem
- Why token co-occurrence clusters correspond to ambiguous contexts
- The n* result: in WikiText-103, non-random PPMI structure begins at n=10 tokens
  (table: n=[5,10,15,20,30,40,50], Z-scores, regime)
- Triangle excess: all 18/18 experiments Z >> 0 (table from task2b_sparse_bound.json)
- Explicit caveat: n*=10 is corpus-specific, not a universal constant

### 5. Empirical Evidence (700 words)
- Task setup: GPT-2, TruthfulQA, WikiText-103 PPMI graph
- Aggregate RHI: null result — triangle presence is too coarse (p=0.715)
- Why null is expected: aggregate fluent text has high triangle base rate
- Per-category: Proverbs p=0.044 (Table: 18 hallucinated, 85.7% with triangle;
  11 correct, 18.2% with triangle)
- Proverbs triangle examples: strike-lightning-strikes (PPMI_mean=5.503),
  know-think-happen (local_cooc=0)
- Why Proverbs: cultural collocations → PPMI-dense clusters → ambiguous contexts
- Theory connection: these triangles are witnesses of contexts where
  sim_correct ≈ sim_incorrect

### 6. Discussion (500 words)
- Limitations: GPT-2 not SOTA; TruthfulQA is specific; ρ is hard to estimate
- What would strengthen: Theorem 2 (converse — motifs predict floor magnitude)
- Open problem: estimate α_n and ε_n for modern LLMs
- The nonlocal null result: why correct answers have more nonlocal triangles
  (correct answers use complex global associations, not local clichés)

### 7. Conclusion (200 words)
- We have established: floor exists (Theorem 1), structural basis for mechanism
  (n*=10, Z>>0), category-level evidence (Proverbs p=0.044)
- We have NOT established: all hallucinations are unavoidable; triangles → hallucination

### References
Include:
- Rödl & Ruciński (1993)
- Bollobás (2001)
- TruthfulQA paper (Lin et al., 2022)
- WikiText-103 (Merity et al., 2016)
- Le Cam (1973) for TV distance lower bound / Fano's inequality
- GPT-2 (Radford et al., 2019)
- sentence-transformers / all-MiniLM-L6-v2 (Reimers & Gurevych, 2019)

---

## Implementation Notes

- `src/theory.py` should have no external dependencies beyond `math` and `numpy`.
- The estimation script can use `json`, `numpy`, `pathlib`, `argparse`.
- The paper rewrite should use the existing docx-generation infrastructure
  (check how paper.js or the existing docx skill works, then match the pattern).
- All result files should be written to `results/` directory.
- After implementing, run:
  ```bash
  python -c "from src.theory import *; print(rare_evidence_bound(0.01, 100))"
  python scripts/run_theory_estimation.py \
    --semantic-results results/task3_rhi_gpt2_semantic.json \
    --category-results results/task3_rhi_by_category.json \
    --output-dir results/
  ```
  and paste the output here.

## Git

After all three tasks are implemented and tests pass:
```bash
git add src/theory.py scripts/run_theory_estimation.py
git commit -m "Add information-theoretic floor bounds (src/theory.py) and estimation script"
```
Do NOT commit the paper or result JSON files.
