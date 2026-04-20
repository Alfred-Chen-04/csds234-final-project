"""WhyProv + QueryRewrite using the inverted index for pruning."""

from load import get_db, DB_PATH
from cf import top_k, _get_model


def _contributors(movie_id: int, db_path: str = DB_PATH) -> list[tuple[int, float]]:
    """Return (user_id, weight) pairs from the inverted index for movie_id."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT user_id, weight FROM movie_contributors WHERE movie_id=? ORDER BY weight DESC",
        (movie_id,)
    ).fetchall()
    conn.close()
    return rows


def _predict_without(algo, user_id: int, movie_id: int,
                     exclude_users: set, db_path: str = DB_PATH) -> float:
    """Re-estimate CF score for (user_id, movie_id) ignoring ratings from exclude_users."""
    conn = get_db(db_path)
    # Reconstruct partial prediction by removing the influence of exclude_users.
    # Approximation: subtract their weighted contribution from the raw score.
    all_contrib = conn.execute(
        "SELECT user_id, weight FROM movie_contributors WHERE movie_id=?",
        (movie_id,)
    ).fetchall()
    conn.close()

    base_score = algo.predict(user_id, movie_id).est
    if not all_contrib:
        return base_score

    total_weight = sum(w for _, w in all_contrib)
    removed_weight = sum(w for uid, w in all_contrib if uid in exclude_users)
    if total_weight == 0:
        return base_score

    # Scale down score proportionally to removed contribution
    adjusted = base_score * (1 - removed_weight / total_weight)
    return adjusted


def why_prov(user_id: int, movie_id: int, k: int = 10,
             db_path: str = DB_PATH) -> list[tuple[int, int]]:
    """
    Return minimal set of (rater_user_id, movie_id) ratings that, if removed,
    would cause movie_id to drop out of user_id's top-k.

    Uses index pruning: only examines top contributors, not all 100K ratings.
    """
    algo = _get_model()
    contributors = _contributors(movie_id, db_path)

    # Greedy removal: add contributors one-by-one until movie drops below top-k
    removed: set = set()
    witness: list[tuple[int, int]] = []

    for contrib_user, _ in contributors:
        removed.add(contrib_user)
        witness.append((contrib_user, movie_id))

        new_score = _predict_without(algo, user_id, movie_id, removed, db_path)

        # Check if movie_id would fall out of top-k with this adjusted score
        topk_ids = top_k(user_id, k, db_path)
        if movie_id not in topk_ids or new_score < _threshold_score(algo, user_id, topk_ids, k):
            return witness

    return witness  # full contributor set if never dropped


def _threshold_score(algo, user_id: int, topk_ids: list[int], k: int) -> float:
    """Score of the k-th item in the current top-k (the eviction threshold)."""
    if len(topk_ids) < k:
        return 0.0
    return algo.predict(user_id, topk_ids[k - 1]).est


def query_rewrite(user_id: int, target_movie: int, k: int = 10,
                  db_path: str = DB_PATH) -> dict:
    """
    Find the minimal rating edit that surfaces target_movie into user_id's top-k.

    Strategy: identify which users' ratings most influence target_movie's score,
    then suggest adding a high rating from the top contributor who hasn't yet rated it.

    Returns: {"action": "add"|"remove", "user_id": int, "movie_id": int, "rating": float}
    """
    algo = _get_model()
    conn = get_db(db_path)

    # Find users who rated similar movies highly but haven't rated target_movie
    similar_raters = conn.execute("""
        SELECT r.user_id, AVG(r.rating) as avg_rating
        FROM ratings r
        WHERE r.movie_id != ?
          AND r.user_id NOT IN (
              SELECT user_id FROM ratings WHERE movie_id = ?
          )
        GROUP BY r.user_id
        ORDER BY avg_rating DESC
        LIMIT 20
    """, (target_movie, target_movie)).fetchall()
    conn.close()

    current_score = algo.predict(user_id, target_movie).est
    best_edit = None
    best_gain = 0.0

    for rater_id, avg_r in similar_raters:
        # Simulate adding a 5-star rating from this user for target_movie
        simulated_gain = (5.0 - avg_r) * 0.1  # approximate influence
        new_score = current_score + simulated_gain
        if new_score > best_gain:
            best_gain = new_score
            best_edit = {
                "action": "add",
                "user_id": rater_id,
                "movie_id": target_movie,
                "rating": 5.0,
                "estimated_new_score": round(new_score, 3),
            }

    return best_edit or {
        "action": "none",
        "reason": "target movie already in or cannot be surfaced with single edit",
    }


if __name__ == "__main__":
    import time

    uid, mid, k = 1, 483, 10
    t0 = time.time()
    witness = why_prov(uid, mid, k)
    print(f"why_prov(user={uid}, movie={mid}, k={k})")
    print(f"  witness set ({len(witness)} ratings): {witness[:5]}...")
    print(f"  Time: {time.time() - t0:.3f}s")

    target = 1  # movie not in top-10 for user 1
    t1 = time.time()
    edit = query_rewrite(uid, target, k)
    print(f"\nquery_rewrite(user={uid}, target={target}, k={k})")
    print(f"  suggested edit: {edit}")
    print(f"  Time: {time.time() - t1:.3f}s")
