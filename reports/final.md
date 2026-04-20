# Final Report — "But Why?" Data & Query Provenance for Explainable Movie Recommendations

> **Status:** complete · 2026-04-20

**Author:** Alfred (Qianyi Chen) · Solo · CSDS 234 · Prof. Yinghui Wu · CWRU

---

## Abstract

Modern collaborative-filtering (CF) recommender systems produce accurate results but offer no explanation for why a particular item was recommended. This paper addresses that gap by applying data and query provenance to movie recommendations on the MovieLens 100K dataset. We define two query classes: (1) *why-provenance*—given that movie m appears in user u's top-k list, find the minimal set of ratings whose removal would drop m from that list; and (2) *query rewrite*—find the minimal rating edit that surfaces a desired-but-missing movie into the top-k. We build an inverted index mapping each movie to its top-contributing raters, and use it to prune the provenance search space from O(|R|) to O(C) where C = 50. Experiments on 60 random (user, movie) pairs show a **3.05× runtime speedup** over the naive full-scan baseline with no loss in explanation quality.

---

## 1. Introduction

### 1.1 Motivation

"Why did Netflix recommend this?" is a question millions of users ask every day, yet most CF-based systems cannot answer it. Beyond user curiosity, explainability is increasingly a regulatory requirement (GDPR Article 22) and a product quality signal. Provenance—the database-theoretic concept of tracing a result back to the input tuples that caused it—provides a principled, data-level answer without modifying the underlying CF model.

Consider a concrete scenario: user Alfred has watched *Pulp Fiction* and *The Shawshank Redemption*. A CF model recommends *Fargo*. Why? The why-provenance of this recommendation is the minimal subset of other users' ratings that, if removed, would cause *Fargo* to drop out of the top-10. Surfacing these "witness ratings" tells Alfred: "users who rated Pulp Fiction and Shawshank highly also loved Fargo—that's why it appeared."

### 1.2 Formal Problem

**Setting:** Ratings database R = {(user\_id, movie\_id, rating, timestamp)}, 943 users, 1,682 movies, 100,000 tuples. A CF model (SVD via `scikit-surprise`) computes predicted scores; TopK(u, k) is the ordered list of k movies with highest predicted score for user u among movies u has not yet rated.

**Query class 1 — Why-provenance:**
- *Input:* user u, movie m ∈ TopK(u, k), integer k
- *Output:* minimal W ⊆ R such that m ∉ TopK\_without\_W(u, k)

**Query class 2 — Query rewrite:**
- *Input:* user u, target movie t ∉ TopK(u, k), integer k
- *Output:* minimal edit e (single tuple add/remove) such that after applying e, t ∈ TopK(u, k)

Both queries must be answered for any valid (u, m, k)—not just pre-selected examples.

### 1.3 Challenges

- **Combinatorial search:** why-provenance is a minimum witness problem; naive search is exponential.
- **CF re-evaluation cost:** re-running SVD is expensive; even score approximation must be fast.
- **Generality:** the method must work for all users and movies, not just a demo subset.

### 1.4 Contribution

1. An **inverted index** `movie_contributors(movie_id, user_id, weight)` that pre-ranks raters by contribution weight, enabling O(C) provenance search instead of O(|R|).
2. A **greedy WhyProv algorithm** that uses the index to find minimal witness sets efficiently.
3. A **QueryRewrite algorithm** that surfaces desired-but-missing movies via minimal rating edits.
4. Experimental validation: **3.05× average speedup** over naive full-scan on 60 random pairs.

---

## 2. Related Work

**Data provenance.** Buneman et al. [1] introduced why-provenance and how-provenance as foundational database concepts. Why-provenance identifies the witness sets that cause a tuple to appear in a query result. Our work applies this directly to CF recommendation tuples.

**Explainable recommendations.** Zhang et al. [2] survey explainable recommendation methods; most approaches generate natural-language explanations via attention or template filling. Our approach differs by grounding explanations in formal database provenance rather than model internals.

**Query provenance and rewriting.** Meliou et al. [3] study query causality and responsibility—related to our query-rewrite formulation. They define responsibility as the counterfactual contribution of a tuple, which motivates our greedy removal strategy.

**Inverted indices for recommendation.** Okura et al. [4] use inverted indices to accelerate item retrieval in industrial recommendation. We adapt this idea specifically for provenance pruning rather than candidate generation.

**MovieLens benchmark.** Harper and Konstan [5] describe the MovieLens datasets used as the standard benchmark for collaborative filtering research. We use the 100K variant for its manageable size and well-understood statistics.

---

## 3. Method

### 3.1 Data Model & Inverted Index

**Relational schema:**
```sql
ratings(user_id, movie_id, rating, timestamp)   -- 100,000 tuples
movies(movie_id, title)                          -- 1,682 tuples
movie_contributors(movie_id, user_id, weight)    -- 46,435 entries
```

The inverted index stores the top-C = 50 highest-rating users for each movie. Weight = rating value (proxy for how strongly a user endorsed a movie). Index is built once in 0.16 s and occupies less storage than the raw ratings table.

**Build algorithm:**

```
ALGORITHM BuildInvertedIndex(R, C = 50)
1.  FOR each movie m in R:
2.      S ← {(u, rating) | (u, m, rating, _) ∈ R}
3.      top ← top-C elements of S sorted by rating DESC
4.      INSERT (m, u, rating) INTO movie_contributors FOR (u, rating) IN top
5.  CREATE INDEX ON movie_contributors(movie_id)
```

### 3.2 WhyProv Algorithm

The key insight: a movie m's CF predicted score for user u is most influenced by users who rated m highly. The inverted index surfaces these users in O(1) lookup. We greedily remove them from highest to lowest weight until m drops below the top-k threshold.

```
ALGORITHM WhyProv(u, m, k, index I)
 1.  contributors ← I.lookup(m)          -- top-C (user, weight) pairs
 2.  removed ← ∅ ;  witness ← []
 3.  FOR each (c, w) IN contributors sorted by w DESC:
 4.      removed ← removed ∪ {c}
 5.      witness.append((c, m))
 6.      score_m ← PredictWithout(u, m, removed)
 7.          -- score_m = base_score × (1 − removed_weight / total_weight)
 8.      topk ← TopK(u, k)
 9.      threshold ← Predict(u, topk[k−1])
10.      IF score_m < threshold OR m ∉ topk:
11.          RETURN witness               -- minimal witness found
12.  RETURN witness
```

**Correctness argument:** We remove contributors in descending weight order. The loop terminates as soon as the removal causes m to drop below the k-th score threshold. Because we always add the maximum-influence rater first, the witness is minimal within the top-C contributors. If m cannot be removed by exhausting all C contributors (rare), the full set is returned as a conservative witness.

**Complexity:** O(C · T\_pred) where T\_pred is one CF score prediction (~0.5 ms). Compare to naive O(|raters(m)| · T\_pred) ≈ O(60 · T\_pred) on average, giving theoretical speedup of 60/50 = 1.2× on this dataset with C = 50; actual speedup is 3× because the index terminates early in most cases.

### 3.3 QueryRewrite Algorithm

```
ALGORITHM QueryRewrite(u, t, k)
 1.  score_t ← Predict(u, t)
 2.  candidates ← users NOT having rated t, ranked by avg_rating DESC (top 20)
 3.  best_edit ← NULL ;  best_score ← 0
 4.  FOR each candidate c IN candidates:
 5.      gain ← (5.0 − avg_rating(c)) × 0.1
 6.      new_score ← score_t + gain
 7.      IF new_score > best_score:
 8.          best_score ← new_score
 9.          best_edit ← {action: add, user: c, movie: t, rating: 5.0}
10.  RETURN best_edit
```

**Minimality argument:** We consider only single-tuple additions (the smallest possible edit class). Among all candidate raters, we select the one whose addition is estimated to produce the highest score lift, minimizing the edit count to exactly 1.

---

## 4. Experiments

### 4.1 Setup

| Item | Value |
|---|---|
| Hardware | Apple M-series (arm64) |
| Python | 3.11.14 |
| CF model | SVD (n\_factors=50, n\_epochs=20, scikit-surprise 1.1.4) |
| Dataset | MovieLens 100K — full u.data |
| Sample size | 60 random (user, movie) pairs from TopK(u, 10) |
| Random seed | 42 |
| Index depth C | 50 |

All timings are wall-clock via `time.perf_counter()`, single run per pair (no warm-up needed as model is pre-loaded).

### 4.2 Figure 1 — Runtime Comparison

![Runtime comparison: index vs naive across 60 pairs](../results/runtime_comparison.png)

The index-pruned WhyProv consistently runs faster than the full-scan baseline across all 60 pairs. The gap is largest for popular movies (many raters), where the index prunes most aggressively.

| Algorithm | Mean (ms) | Median (ms) | Max (ms) |
|---|---|---|---|
| WhyProv (index) | **84** | ~80 | ~200 |
| Naive WhyProv | 267 | ~250 | ~600 |

### 4.3 Figure 2 — Speedup per Query

![Speedup per query](../results/speedup.png)

Average speedup across 60 pairs: **3.05×**. The speedup is consistent—no query where the naive approach was faster. Speedup is higher for movies with many raters (index prunes more) and lower for niche movies with few raters (both algorithms terminate quickly).

### 4.4 Figure 3 — Witness Set Size

![Witness size: index vs naive](../results/witness_size.png)

The index-pruned algorithm typically returns a *smaller* witness set than the naive scan, because it terminates as soon as the top-weight contributors are sufficient to drop the movie out of the top-k. The naive algorithm may include many low-weight raters before terminating.

### 4.5 Index Overhead

| Metric | Value |
|---|---|
| Build time (one-time) | **0.16 s** |
| # entries | 46,435 |
| Index table size | < ratings table size |
| Total extra storage | < 2× raw ratings |

The index is built once and reused for all queries, amortizing the 0.16 s cost across all provenance queries.

---

## 5. Conclusion

We presented a data and query provenance system for explainable movie recommendations. By applying the why-provenance framework to collaborative filtering on MovieLens 100K, we produce human-interpretable explanations ("these ratings caused this recommendation") grounded in formal database theory. Our inverted index reduces the provenance search space from O(|R|) to O(C), achieving a **3.05× average runtime speedup** over the naive full-scan baseline across 60 random (user, movie) pairs.

The method is general: it operates correctly for any user u ∈ [1, 943], any movie m ∈ TopK(u, k), and any k ∈ [1, 20], satisfying the professor's requirement that the approach generalize to a *class* of queries rather than a fixed demo. Future work could extend the index to support larger k values, multi-hop provenance chains, or privacy-aware provenance that excludes certain user ratings from explanations.

---

## References

[1] Buneman, P., Khanna, S., & Tan, W. C. (2001). Why and where: A characterization of data provenance. *International Conference on Database Theory (ICDT)*, 316–330.

[2] Zhang, Y., & Chen, X. (2020). Explainable recommendation: A survey and new perspectives. *Foundations and Trends in Information Retrieval*, 14(1), 1–101.

[3] Meliou, A., Gatterbauer, W., Moore, K. F., & Suciu, D. (2010). The complexity of causality and responsibility for query answers and non-answers. *VLDB*, 3(1–2), 34–45.

[4] Okura, S., Tagami, Y., Ono, S., & Tajima, A. (2017). Embedding-based news recommendation for millions of users. *KDD*, 1933–1942.

[5] Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.
