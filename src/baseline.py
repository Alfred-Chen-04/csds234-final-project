"""Naive full-scan baselines — no index, scans all ratings."""

from load import get_db, DB_PATH
from cf import top_k, _get_model


def _all_raters(movie_id: int, db_path: str = DB_PATH) -> list[tuple[int, float]]:
    """Return ALL (user_id, rating) pairs for movie_id — no pruning."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT user_id, rating FROM ratings WHERE movie_id=? ORDER BY rating DESC",
        (movie_id,)
    ).fetchall()
    conn.close()
    return rows


def _predict_without_naive(algo, user_id: int, movie_id: int,
                           exclude_users: set, db_path: str = DB_PATH) -> float:
    """Same score adjustment as provenance.py but using full rating scan."""
    conn = get_db(db_path)
    all_ratings = conn.execute(
        "SELECT user_id, rating FROM ratings WHERE movie_id=?",
        (movie_id,)
    ).fetchall()
    conn.close()

    base_score = algo.predict(user_id, movie_id).est
    total_weight = sum(r for _, r in all_ratings)
    removed_weight = sum(r for uid, r in all_ratings if uid in exclude_users)
    if total_weight == 0:
        return base_score
    return base_score * (1 - removed_weight / total_weight)


def naive_why_prov(user_id: int, movie_id: int, k: int = 10,
                   db_path: str = DB_PATH) -> list[tuple[int, int]]:
    """
    Full-scan why-provenance: iterates ALL raters of movie_id (not just index top-C).
    Same greedy logic as why_prov() but without index pruning.
    """
    algo = _get_model()
    raters = _all_raters(movie_id, db_path)

    removed: set = set()
    witness: list[tuple[int, int]] = []

    for rater_id, _ in raters:
        removed.add(rater_id)
        witness.append((rater_id, movie_id))
        new_score = _predict_without_naive(algo, user_id, movie_id, removed, db_path)
        topk_ids = top_k(user_id, k, db_path)
        if movie_id not in topk_ids or new_score < algo.predict(user_id, topk_ids[k - 1]).est:
            return witness

    return witness


def naive_query_rewrite(user_id: int, target_movie: int,
                        db_path: str = DB_PATH) -> dict:
    """
    Full-scan query rewrite: scans ALL users not having rated target_movie.
    Same logic as query_rewrite() but without any pruning.
    """
    algo = _get_model()
    conn = get_db(db_path)
    all_users = conn.execute("""
        SELECT DISTINCT user_id FROM ratings
        WHERE user_id NOT IN (
            SELECT user_id FROM ratings WHERE movie_id = ?
        )
    """, (target_movie,)).fetchall()
    conn.close()

    current_score = algo.predict(user_id, target_movie).est
    best_edit = None
    best_score = 0.0

    for (rater_id,) in all_users:
        simulated_score = current_score + 0.1  # uniform approximation, no index weight
        if simulated_score > best_score:
            best_score = simulated_score
            best_edit = {
                "action": "add",
                "user_id": rater_id,
                "movie_id": target_movie,
                "rating": 5.0,
                "estimated_new_score": round(simulated_score, 3),
            }

    return best_edit or {"action": "none"}


if __name__ == "__main__":
    import time

    uid, mid, k = 1, 483, 10

    t0 = time.time()
    w = naive_why_prov(uid, mid, k)
    print(f"naive_why_prov(user={uid}, movie={mid}, k={k})")
    print(f"  witness ({len(w)} ratings): {w[:5]}...")
    print(f"  Time: {time.time() - t0:.3f}s")

    t1 = time.time()
    e = naive_query_rewrite(uid, mid)
    print(f"\nnaive_query_rewrite(user={uid}, target={mid}, k={k})")
    print(f"  edit: {e}")
    print(f"  Time: {time.time() - t1:.3f}s")
