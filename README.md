# But Why? — Data & Query Provenance for Explainable Movie Recommendations

**CSDS 234 Final Project** · Prof. Yinghui Wu · CWRU · Spring 2026
**Author:** Alfred (Qianyi Chen) · Solo

---

## Motivation

When Netflix recommends *Inception*, two questions are interesting:

1. **"Why did you recommend it?"** — which of my past ratings *caused* this recommendation?
2. **"I wanted *Spirited Away* instead — how do I get it?"** — what is the *smallest* change to the recommendation query that would surface it?

These are the classic **Data Provenance** and **Query Provenance** problems, applied to a top-K collaborative-filtering recommender.

---

## Contribution

1. A formal definition of both provenance problems for top-K CF recommendations.
2. An **inverted index** over `(movie → top contributing users)` that enables sub-linear-time why-provenance.
3. Two algorithms — `WhyProv` and `QueryRewrite` — with pseudo-code.
4. An experimental study on **MovieLens 100K** showing ≥5× speedup and ≥50% smaller provenance versus naive baselines, across three factors (K, user activity, subsample size).

---

## Dataset

[MovieLens 100K](https://files.grouplens.org/datasets/movielens/ml-100k.zip) — 100,000 ratings from 943 users on 1,682 movies. Standard benchmark for CF research.

---

## Stack

Python 3.11 · SQLite · [scikit-surprise](https://surpriselib.com/) · numpy · matplotlib

---

## Quick Start

```bash
pip install scikit-surprise numpy matplotlib

python src/load.py          # download + load data
python src/index.py         # build inverted index
python src/experiments.py   # run experiments, write figures to results/
```

---

## Repository Layout

```
src/                    Python source
  load.py               download + load MovieLens 100K into SQLite
  index.py              build movie -> top-C contributors index
  cf.py                 SVD recommender wrapper (scikit-surprise)
  provenance.py         WhyProv + QueryRewrite (index-pruned)
  baseline.py           naive full-scan baselines
  experiments.py        runtime / k-sweep / popularity experiments

results/                experiment outputs (figures + CSVs)
reports/                milestone + final reports (Markdown)
CSDS234_Qianyi_slides.pptx   final presentation
```

---

## Reports

- [M1 — Project Statement](reports/M1.md)
- [M2 — Dataset & Index](reports/M2.md)
- [M3 — Algorithms & Experimental Plan](reports/M3.md)
- [Final Report](reports/final.md)
