"""Core contribution: WhyProv + QueryRewrite. Implemented in M3."""


def why_prov(user_id: int, movie_id: int, k: int) -> list[tuple[int, int]]:
    """Minimum subset of (user, movie) tuples explaining why movie_id is in top-k for user_id."""
    raise NotImplementedError("M3 deliverable")


def query_rewrite(user_id: int, target_movie: int, query: dict) -> dict:
    """Minimum edit to `query` that surfaces target_movie while preserving top-k."""
    raise NotImplementedError("M3 deliverable")
