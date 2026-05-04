"""Run all experiments and emit figures + CSVs into results/."""

import os
import time
import random
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load import get_db, DB_PATH
from cf import top_k
from provenance import why_prov
from baseline import naive_why_prov

RESULTS_DIR = "results"
N_SAMPLES = 60   # pairs to evaluate (satisfies >=50 requirement)
K = 10
RANDOM_SEED = 42


def _sample_pairs(n: int, db_path: str = DB_PATH,
                  pool_k: int = 20) -> list[tuple[int, int]]:
    """Sample n random (user, movie) pairs where movie is randomly drawn from
    the user's top-`pool_k` recommendations. Drawing from top-20 (rather than
    only top-1) gives realistic coverage of what users would actually see in a
    recommendation UI, and exposes the algorithm to varying movie popularity."""
    conn = get_db(db_path)
    users = [r[0] for r in conn.execute(
        "SELECT DISTINCT user_id FROM ratings ORDER BY RANDOM() LIMIT ?", (n * 2,)
    ).fetchall()]
    conn.close()

    pairs = []
    for uid in users:
        topk = top_k(uid, pool_k)
        if topk:
            mid = random.choice(topk)
            pairs.append((uid, mid))
        if len(pairs) >= n:
            break
    return pairs[:n]


def run_timing_experiment(pairs: list[tuple[int, int]]) -> tuple[list, list]:
    """Return (index_times, naive_times) for why_prov vs naive_why_prov."""
    index_times, naive_times = [], []
    for uid, mid in pairs:
        t0 = time.perf_counter()
        why_prov(uid, mid, K)
        index_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        naive_why_prov(uid, mid, K)
        naive_times.append(time.perf_counter() - t0)

    return index_times, naive_times


def run_witness_size_experiment(pairs: list[tuple[int, int]]) -> tuple[list, list]:
    """Return (index_sizes, naive_sizes) — number of ratings in each witness set."""
    index_sizes, naive_sizes = [], []
    for uid, mid in pairs:
        index_sizes.append(len(why_prov(uid, mid, K)))
        naive_sizes.append(len(naive_why_prov(uid, mid, K)))
    return index_sizes, naive_sizes


def plot_runtime(index_times, naive_times) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(1, len(index_times) + 1)
    ax.plot(x, [t * 1000 for t in naive_times], label="Naive (full scan)", color="tomato")
    ax.plot(x, [t * 1000 for t in index_times], label="Index-pruned", color="steelblue")
    ax.set_xlabel("Query pair (sorted by naive time)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"WhyProv runtime: index vs naive ({N_SAMPLES} random pairs, k={K})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "runtime_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved results/runtime_comparison.png")


def plot_speedup(index_times, naive_times) -> None:
    speedups = [n / i if i > 0 else 0 for n, i in zip(naive_times, index_times)]
    avg = sum(speedups) / len(speedups)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(speedups) + 1), speedups, color="steelblue", alpha=0.7)
    ax.axhline(avg, color="red", linestyle="--", label=f"Mean speedup: {avg:.1f}×")
    ax.set_xlabel("Query pair")
    ax.set_ylabel("Speedup (naive / index)")
    ax.set_title(f"Speedup per query (n={N_SAMPLES})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speedup.png"), dpi=150)
    plt.close(fig)
    print(f"Saved results/speedup.png  (avg speedup: {avg:.2f}×)")


def plot_witness_size(index_sizes, naive_sizes) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(naive_sizes, index_sizes, alpha=0.6, color="steelblue")
    ax.plot([0, max(naive_sizes)], [0, max(naive_sizes)], "r--", label="y=x")
    ax.set_xlabel("Naive witness size")
    ax.set_ylabel("Index witness size")
    ax.set_title("Witness set size: index vs naive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "witness_size.png"), dpi=150)
    plt.close(fig)
    print("Saved results/witness_size.png")


def save_csv(pairs, index_times, naive_times, index_sizes, naive_sizes) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "movie_id", "index_ms", "naive_ms", "speedup",
                    "index_witness", "naive_witness"])
        for (uid, mid), it, nt, isz, nsz in zip(
                pairs, index_times, naive_times, index_sizes, naive_sizes):
            w.writerow([uid, mid, round(it * 1000, 2), round(nt * 1000, 2),
                        round(nt / it if it > 0 else 0, 2), isz, nsz])
    print(f"Saved {path}")


def run_k_sweep(pairs: list[tuple[int, int]],
                k_values: list[int] = [5, 10, 20]) -> dict:
    """For each k, time index vs naive across all pairs. Return {k: (index_mean_ms, naive_mean_ms, speedup)}."""
    results: dict = {}
    for k in k_values:
        idx_times, naive_times = [], []
        for uid, mid in pairs:
            t0 = time.perf_counter()
            why_prov(uid, mid, k)
            idx_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            naive_why_prov(uid, mid, k)
            naive_times.append(time.perf_counter() - t0)

        idx_mean = sum(idx_times) / len(idx_times) * 1000
        naive_mean = sum(naive_times) / len(naive_times) * 1000
        speedup = naive_mean / idx_mean if idx_mean > 0 else 0
        results[k] = (idx_mean, naive_mean, speedup)
        print(f"  k={k}: index {idx_mean:.1f}ms | naive {naive_mean:.1f}ms | speedup {speedup:.2f}×")
    return results


def plot_k_sweep(results: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ks = sorted(results.keys())
    idx_means = [results[k][0] for k in ks]
    naive_means = [results[k][1] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(ks))
    width = 0.35
    ax.bar([i - width / 2 for i in x], naive_means, width, label="Naive (full scan)", color="tomato")
    ax.bar([i + width / 2 for i in x], idx_means, width, label="Index-pruned", color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Mean runtime per query (ms)")
    ax.set_title(f"WhyProv runtime vs k (n={len(ks) and N_SAMPLES} pairs)")
    for i, k in enumerate(ks):
        ax.text(i, max(idx_means[i], naive_means[i]) + 5,
                f"{results[k][2]:.2f}×", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "runtime_vs_k.png"), dpi=150)
    plt.close(fig)
    print("Saved results/runtime_vs_k.png")


def save_k_sweep_csv(results: dict) -> None:
    path = os.path.join(RESULTS_DIR, "k_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "index_mean_ms", "naive_mean_ms", "speedup"])
        for k in sorted(results.keys()):
            idx, naive, sp = results[k]
            w.writerow([k, round(idx, 2), round(naive, 2), round(sp, 2)])
    print(f"Saved {path}")


def _movie_popularity(db_path: str = DB_PATH) -> dict:
    """Return {movie_id: rater_count}."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT movie_id, COUNT(*) FROM ratings GROUP BY movie_id"
    ).fetchall()
    conn.close()
    return {mid: cnt for mid, cnt in rows}


def run_popularity_stratification(pairs: list[tuple[int, int]],
                                  index_times: list, naive_times: list) -> dict:
    """Bucket pairs by movie rater count: cold (<=20), medium (21-50), hot (51+)."""
    pop = _movie_popularity()
    buckets = {"cold (≤20)": [], "medium (21–50)": [], "hot (51+)": []}

    for (_uid, mid), it, nt in zip(pairs, index_times, naive_times):
        n = pop.get(mid, 0)
        speedup = nt / it if it > 0 else 0
        if n <= 20:
            buckets["cold (≤20)"].append((n, speedup, it * 1000, nt * 1000))
        elif n <= 50:
            buckets["medium (21–50)"].append((n, speedup, it * 1000, nt * 1000))
        else:
            buckets["hot (51+)"].append((n, speedup, it * 1000, nt * 1000))

    summary = {}
    for label, entries in buckets.items():
        if entries:
            mean_sp = sum(e[1] for e in entries) / len(entries)
            mean_idx = sum(e[2] for e in entries) / len(entries)
            mean_naive = sum(e[3] for e in entries) / len(entries)
            summary[label] = {
                "n": len(entries),
                "mean_speedup": mean_sp,
                "mean_index_ms": mean_idx,
                "mean_naive_ms": mean_naive,
            }
            print(f"  {label}: n={len(entries)}  speedup={mean_sp:.2f}×  "
                  f"index={mean_idx:.1f}ms  naive={mean_naive:.1f}ms")
        else:
            summary[label] = {"n": 0, "mean_speedup": 0,
                              "mean_index_ms": 0, "mean_naive_ms": 0}
    return summary


def plot_popularity(summary: dict) -> None:
    labels = list(summary.keys())
    speedups = [summary[l]["mean_speedup"] for l in labels]
    counts = [summary[l]["n"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, speedups, color=["#9ecae1", "#4292c6", "#08519c"])
    for i, (b, n) in enumerate(zip(bars, counts)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"n={n}\n{speedups[i]:.2f}×", ha="center", fontsize=9)
    ax.set_ylabel("Mean speedup (naive / index)")
    ax.set_xlabel("Movie popularity (rater count)")
    ax.set_title("Speedup by movie popularity")
    ax.set_ylim(0, max(speedups) * 1.25 if speedups else 1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speedup_by_popularity.png"), dpi=150)
    plt.close(fig)
    print("Saved results/speedup_by_popularity.png")


def save_popularity_csv(summary: dict) -> None:
    path = os.path.join(RESULTS_DIR, "popularity.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n", "mean_speedup", "mean_index_ms", "mean_naive_ms"])
        for label, s in summary.items():
            w.writerow([label, s["n"], round(s["mean_speedup"], 2),
                        round(s["mean_index_ms"], 2), round(s["mean_naive_ms"], 2)])
    print(f"Saved {path}")


def main() -> None:
    random.seed(RANDOM_SEED)
    print(f"Sampling {N_SAMPLES} (user, movie) pairs at k={K}...")
    pairs = _sample_pairs(N_SAMPLES)
    print(f"Got {len(pairs)} pairs.\n")

    print("[1/4] Main timing experiment (k=10)...")
    index_times, naive_times = run_timing_experiment(pairs)
    index_sizes, naive_sizes = run_witness_size_experiment(pairs)

    plot_runtime(index_times, naive_times)
    plot_speedup(index_times, naive_times)
    plot_witness_size(index_sizes, naive_sizes)
    save_csv(pairs, index_times, naive_times, index_sizes, naive_sizes)

    avg_speedup = sum(n / i for n, i in zip(naive_times, index_times) if i > 0) / len(pairs)
    print(f"  avg speedup = {avg_speedup:.2f}×\n")

    print("[2/4] k-sweep experiment (k ∈ {5, 10, 20})...")
    k_results = run_k_sweep(pairs)
    plot_k_sweep(k_results)
    save_k_sweep_csv(k_results)
    print()

    print("[3/4] Movie-popularity stratification...")
    pop_summary = run_popularity_stratification(pairs, index_times, naive_times)
    plot_popularity(pop_summary)
    save_popularity_csv(pop_summary)
    print()

    print("[4/4] Summary")
    print(f"  Main: avg speedup {avg_speedup:.2f}× across {len(pairs)} pairs at k={K}")
    print(f"  k-sweep: speedup {k_results[5][2]:.2f}× (k=5) → "
          f"{k_results[10][2]:.2f}× (k=10) → {k_results[20][2]:.2f}× (k=20)")


if __name__ == "__main__":
    main()
