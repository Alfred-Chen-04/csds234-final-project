"""CSV -> SQLite loader for MovieLens 100K."""

import sqlite3
import csv
import os

RAW_DIR = "data/raw/ml-100k"
DB_PATH = "data/ratings.db"


def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ratings (
            user_id   INTEGER NOT NULL,
            movie_id  INTEGER NOT NULL,
            rating    REAL    NOT NULL,
            timestamp INTEGER NOT NULL,
            PRIMARY KEY (user_id, movie_id)
        );
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            title    TEXT NOT NULL
        );
    """)


def load_ratings(conn: sqlite3.Connection) -> int:
    path = os.path.join(RAW_DIR, "u.data")
    rows = []
    with open(path, newline="") as f:
        for line in f:
            uid, mid, r, ts = line.strip().split("\t")
            rows.append((int(uid), int(mid), float(r), int(ts)))
    conn.executemany(
        "INSERT OR REPLACE INTO ratings VALUES (?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


def load_movies(conn: sqlite3.Connection) -> int:
    path = os.path.join(RAW_DIR, "u.item")
    rows = []
    with open(path, newline="", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            rows.append((int(row[0]), row[1]))
    conn.executemany(
        "INSERT OR REPLACE INTO movies VALUES (?,?)", rows
    )
    conn.commit()
    return len(rows)


def main() -> None:
    conn = get_db()
    create_schema(conn)
    n_ratings = load_ratings(conn)
    n_movies = load_movies(conn)
    conn.close()
    print(f"Loaded {n_ratings} ratings and {n_movies} movies into {DB_PATH}")


if __name__ == "__main__":
    main()
