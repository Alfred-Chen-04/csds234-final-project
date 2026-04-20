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


def _sample_pairs(n: int, db_path: str = DB_PATH) -> list[tuple[int, int]]:
    """Sample n random (user_id, movie_id) pairs where movie is in user's top-k."""
    conn = get_db(db_path)
    users = [r[0] for r in conn.execute(
        "SELECT DISTINCT user_id FROM ratings ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()]
    conn.close()

    pairs = []
    for uid in users:
        topk = top_k(uid, K)
        if topk:
            pairs.append((uid, topk[0]))
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


def main() -> None:
    random.seed(RANDOM_SEED)
    print(f"Sampling {N_SAMPLES} (user, movie) pairs...")
    pairs = _sample_pairs(N_SAMPLES)
    print(f"Got {len(pairs)} pairs. Running experiments...")

    index_times, naive_times = run_timing_experiment(pairs)
    index_sizes, naive_sizes = run_witness_size_experiment(pairs)

    plot_runtime(index_times, naive_times)
    plot_speedup(index_times, naive_times)
    plot_witness_size(index_sizes, naive_sizes)
    save_csv(pairs, index_times, naive_times, index_sizes, naive_sizes)

    avg_speedup = sum(n / i for n, i in zip(naive_times, index_times) if i > 0) / len(pairs)
    print(f"\nSummary: avg speedup = {avg_speedup:.2f}×  "
          f"| index avg {sum(index_times)/len(index_times)*1000:.1f}ms  "
          f"| naive avg {sum(naive_times)/len(naive_times)*1000:.1f}ms")


if __name__ == "__main__":
    main()
