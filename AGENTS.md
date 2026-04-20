# AGENTS.md — AI Collaboration Rules

> **Read this file first in every new session.** It defines what this project is, who the user is, and what you must (and must NOT) do.

---

## 1. Project One-Liner

**CSDS 234 Final Project: "But Why?" — Data & Query Provenance for Explainable Movie Recommendations.**

Given a top-K recommendation, compute (a) the minimum set of ratings that explain it, and (b) the minimum query edit that surfaces a desired-but-missing movie. Contribution = inverted-index pruning algorithm + experimental comparison against naive baselines on MovieLens 100K.

This is a **course project**, not production software. The deliverable is **4 milestone reports + 1 final report + 1 presentation**, not a deployed system.

---

## 2. User Profile

- **Name:** Alfred (Qianyi Chen)
- **Course:** CSDS 234 (Structured and Unstructured Data), Prof. Yinghui Wu, CWRU
- **Career target:** AI Product Manager (this project doubles as portfolio material)
- **Programming skill:** weak. Knows basic DB concepts and simple algorithms. Cannot debug complex Python.
- **Team:** solo
- **Primary goal:** **full marks first, novelty second.** Do not sacrifice score for ambition.
- **Language:** communicate in **Chinese (简体中文)** by default. Code, filenames, and reports in English.

---

## 3. Hard Rules (Red Lines)

Violate any of these = wasted work.

1. **Do NOT reimplement collaborative filtering.** Use the `surprise` library. The contribution is provenance + index, not CF.
2. **Do NOT scale up the dataset.** Stay on MovieLens 100K. Do not add 1M/10M/25M "for robustness."
3. **Do NOT add features not in the plan.** No web UI, no LLM integration, no privacy layer, no cloud.
4. **Do NOT restructure the folders.** The layout in §5 is fixed. New files go into existing folders.
5. **Do NOT write code during M1.** M1 is a planning document only.
6. **Do NOT skip user confirmation before sending emails, changing scope, or deleting work.**
7. **Code stays small (~300 LoC total).** If a file exceeds 150 lines, split or simplify.

---

## 4. What Scores Full Marks (grading priorities)

Professor Wu's rubric (from `references/Project Report & Milestones.pdf`):

| Section | Weight | What it means here |
|---|---|---|
| Problem | 3% | Clear formal input/output definition |
| Related Work | 2% | 3–5 refs on provenance + explainable rec |
| **Method** | **8%** | **Pseudo-code with numbered lines + index design** |
| **Experiments** | **8%** | **Runtime/size/quality curves vs baseline, on ≥3 factors** |
| Presentation | 3% | Story-driven: "why did Netflix recommend this?" → algorithm → speedup bar chart |

**The 16% in Method + Experiments is where the project wins or loses.** Everything else is hygiene.

---

## 5. Folder Layout (fixed)

```
CSDS 234 Final Project/
├── AGENTS.md              ← this file
├── README.md              ← human-facing project summary
├── data/
│   ├── raw/              ← ml-100k extracted here (u.data, u.item, ...)
│   └── ratings.db        ← generated SQLite database
├── src/
│   ├── load.py           ← CSV → SQLite loader
│   ├── cf.py             ← recommendation via `surprise`
│   ├── index.py          ← inverted index builder
│   ├── provenance.py     ← WhyProv + QueryRewrite (core contribution)
│   ├── baseline.py       ← naive full-scan baselines
│   └── experiments.py    ← run all experiments, emit figures
├── results/              ← figures (.png) + tables (.csv)
├── reports/
│   ├── M1.md             ← Project Statement (due early)
│   ├── M2.md             ← Dataset + Index
│   ├── M3.md             ← Algorithms + Experimental Plan
│   └── final.md          ← Final report (5–10 pages)
└── references/           ← course PDFs, screenshots, templates
```

---

## 6. Tech Stack

- **Python 3.11**
- **SQLite** (built into Python, no install)
- **surprise** (`pip install scikit-surprise`) — CF baseline
- **numpy**, **matplotlib** — experiments and figures
- **pandoc** — Markdown → Word for final submission

No other dependencies. No Docker. No cloud.

---

## 7. Milestone Cadence

| Milestone | Output | LoC added | When |
|---|---|---|---|
| M1 | `reports/M1.md` | 0 (docs only) | Week 1 |
| M2 | `reports/M2.md`, `src/load.py`, `src/index.py` | ~100 | Week 2–3 |
| M3 | `reports/M3.md`, `src/provenance.py`, `src/baseline.py` (pseudo-code first, then impl) | ~150 | Week 4–5 |
| Final | `reports/final.md`, `src/experiments.py`, figures in `results/` | ~50 | Week 6–8 |

Report templates are in `references/Project milestone template/`. Follow them section by section.

---

## 8. How to Run (once built)

```bash
# one-time setup
pip install scikit-surprise numpy matplotlib

# load data
python src/load.py       # downloads ml-100k, builds ratings.db

# build index
python src/index.py      # writes inverted index into ratings.db

# run all experiments + generate figures
python src/experiments.py   # outputs into results/
```

Target: fresh checkout → `python src/experiments.py` regenerates all figures in <2 min.

---

## 9. Interaction Rules for the AI

1. **Respond in Chinese.** Code and files in English.
2. **Confirm before destructive actions** (delete, git push, send email).
3. **Before adding any new file,** check §5 — does it belong in an existing folder? If the answer requires a new folder, ask first.
4. **Keep answers short.** Alfred wants decisions, not essays. ≤5 sentences unless explaining an algorithm.
5. **When in doubt about scope,** re-read §3 and §4. Refuse to scope-creep.
6. **Track milestone progress** in `reports/M*.md`, not in chat.

---

## 10. Current Status

- ✅ Plan approved, folder structure scaffolded
- ⏳ Waiting on professor's email reply confirming topic
- ⏳ Next action: if approved, start M1 report

Update this section when status changes.
