---
title: "But Why?"
subtitle: "Data and Query Provenance for Explainable Movie Recommendations"
author: "Qianyi Chen (Alfred) — Solo — CSDS 234 — Prof. Yinghui Wu — CWRU"
date: "May 2026"
---

# Why was this recommended?

Netflix recommends *Fargo* to Alfred. Why? His love for *Pulp Fiction*? His rating of *Shawshank*? Some pattern shared with strangers?

- **Regulatory pressure:** GDPR Article 22 — users have the right to "meaningful information" about automated decisions.
- **Product trust:** explainable recommendations build user confidence.
- **Today's CF models** (matrix factorization, SVD) hide their reasoning in 50-dim latent vectors — black-box by construction.

Most existing methods modify the model (hurting accuracy) or generate post-hoc text (may not be faithful).

**Our angle:** explain via the database, using formal data provenance.

# Two query classes

**1. Why-Provenance**

- Input: user `u`, movie `m` in `TopK(u, k)`
- Output: minimal `W` such that removing `W` drops `m` out of top-k
- *"Which ratings caused this recommendation?"*

**2. Query Rewrite**

- Input: user `u`, target movie `t` not in `TopK(u, k)`
- Output: minimal edit `e` that surfaces `t` into top-k
- *"What change would surface this missing movie?"*

**Generality requirement** (per professor): method must work for any user, any movie, any k in [1, 20] — not just hand-picked demos.

# Inverted index for pruning

**Naive search:** consider all raters of m — up to O(|R|) per query.

**Our index:** precompute, for each movie, the top-C = 50 highest-rating contributors. Search becomes O(C).

**Key insight:** a movie's predicted score is dominated by its strongest raters. If they don't move the needle, the long tail won't either.

| Metric | Value |
|---|---|
| Index build (one-time) | 0.16 s |
| # entries | 46,435 (~28/movie average, capped at 50) |
| Storage | smaller than raw ratings table |

# WhyProv algorithm

```
ALGORITHM WhyProv(u, m, k, index I)

 1.  contributors <- I.lookup(m)            # top-C, O(C) lookup
 2.  removed <- {} ;  witness <- []
 3.  FOR each (c, w) in contributors sorted by w DESC:
 4.      removed <- removed + {c}
 5.      witness.append((c, m))
 6.      score_m <- ScoreDecayApprox(u, m, removed)
 7.      threshold <- Predict(u, TopK(u,k)[k-1])
 8.      IF score_m < threshold OR m not in TopK(u, k):
 9.          RETURN witness                  # minimal!
10.  RETURN witness
```

**Score-decay approximation:** instead of retraining SVD per query (seconds), scale the base score by removed-weight fraction. Rank-monotonic — sufficient for top-k membership decisions.

# Result 1: 1.77× speedup at k=10

![Runtime: index vs naive across 60 random pairs](../results/runtime_comparison.png)

**Setup:** MovieLens 100K · 60 random (user, movie) pairs sampled from each user's top-20 · k = 10 · seed = 42.

| | Index-pruned | Naive |
|---|---|---|
| Mean runtime | **49 ms** | 95 ms |
| Speedup | **1.77×** | — |

In **none** of 60 pairs does naive beat the index. Witness sets are also smaller because the index terminates earlier.

# Result 2: speedup grows with k

![Runtime vs k](../results/runtime_vs_k.png)

| k | Index | Naive | Speedup |
|---|---|---|---|
| 5 | 44 ms | 67 ms | **1.50×** |
| 10 | 49 ms | 95 ms | **1.94×** |
| 20 | 61 ms | 150 ms | **2.45×** |

Speedup grows monotonically. Larger k → naive scan does more per-query work → bigger pruning win. **The method is most valuable for deeper top-k lists.**

# Result 3: speedup vs movie popularity

![Speedup by popularity](../results/speedup_by_popularity.png)

| Bucket | n | Mean speedup |
|---|---|---|
| Medium (21–50 raters) | 3 | 1.38× |
| Hot (51+ raters) | 57 | **1.79×** |

Speedup correlates with movie popularity. Hot movies have a long rater tail to prune — the index pays off most there. Medium movies have fewer raters, so less to prune.

# Limitations & future work

**Limitations (acknowledged):**

- **Score-decay approximation** is rank-monotonic but not the true counterfactual SVD re-evaluation.
- **QueryRewrite** restricted to single-rating additions only.
- Top-k sampling is **popularity-biased** — no cold movies in our 60 pairs.
- Single dataset (MovieLens 100K).

**Future work:**

- Validate score-decay vs. true counterfactual SVD on a small sample.
- Multi-hop provenance — explain the explainers themselves.
- Scale to MovieLens 1M / 25M.
- Human study: are why-provenance explanations more trusted than attention-based ones?
- Integrate with sequence models (SASRec, BERT4Rec).

# Conclusion

**Database provenance, applied.** A formal, faithful, fast way to explain CF recommendations.

| Number | Meaning |
|---|---|
| **1.77×** | speedup at k=10 |
| **2.45×** | speedup at k=20 |
| **0.16 s** | one-time index build |
| **60** | random (user, movie) pairs evaluated |
| **0** | model changes required |

**Code & report:** github.com/Alfred-Chen-04/csds234-final-project

**Thank you. Questions?**
