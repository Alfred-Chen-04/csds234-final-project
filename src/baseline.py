"""Naive full-scan baselines for comparison. Implemented in M3."""


def naive_why_prov(user_id: int, movie_id: int, k: int) -> list[tuple[int, int]]:
    raise NotImplementedError("M3 deliverable")


def naive_query_rewrite(user_id: int, target_movie: int, query: dict) -> dict:
    raise NotImplementedError("M3 deliverable")
