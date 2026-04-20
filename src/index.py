"""Inverted index: movie_id -> top contributing users, stored in SQLite."""

import time
from load import get_db, DB_PATH

TOP_C = 50  # max contributors stored per movie


def build_index(db_path: str = DB_PATH) -> None:
    conn = get_db(db_path)

    conn.executescript("""
        DROP TABLE IF EXISTS movie_contributors;
        CREATE TABLE movie_contributors (
            movie_id INTEGER NOT NULL,
            user_id  INTEGER NOT NULL,
            weight   REAL    NOT NULL,
            PRIMARY KEY (movie_id, user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_mc_movie ON movie_contributors(movie_id);
    """)

    # weight = number of ratings user gave to movies similar to this movie
    # proxy: how many users co-rated the same movies (sum of shared ratings)
    # Simple but general: weight = rating value (higher raters contribute more)
    rows = conn.execute("""
        SELECT movie_id, user_id, rating AS weight
        FROM ratings
    """).fetchall()

    # keep top-C users per movie by weight (rating value)
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for movie_id, user_id, weight in rows:
        buckets[movie_id].append((weight, user_id))

    insert_rows = []
    for movie_id, entries in buckets.items():
        top = sorted(entries, reverse=True)[:TOP_C]
        for weight, user_id in top:
            insert_rows.append((movie_id, user_id, weight))

    conn.executemany(
        "INSERT OR REPLACE INTO movie_contributors VALUES (?,?,?)", insert_rows
    )
    conn.commit()

    n_movies = conn.execute("SELECT COUNT(DISTINCT movie_id) FROM movie_contributors").fetchone()[0]
    n_entries = conn.execute("SELECT COUNT(*) FROM movie_contributors").fetchone()[0]
    conn.close()
    print(f"Index built: {n_movies} movies, {n_entries} entries (top-{TOP_C} per movie)")


if __name__ == "__main__":
    t0 = time.time()
    build_index()
    print(f"Build time: {time.time() - t0:.2f}s")
