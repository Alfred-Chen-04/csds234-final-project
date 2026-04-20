"""Collaborative filtering via surprise SVD. Trains once, caches predictions."""

import os
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from load import DB_PATH, get_db

MODEL_PATH = "data/svd_model.pkl"


def _load_data(db_path: str = DB_PATH):
    conn = get_db(db_path)
    rows = conn.execute("SELECT user_id, movie_id, rating FROM ratings").fetchall()
    conn.close()
    reader = Reader(rating_scale=(1, 5))
    df = pd.DataFrame(rows, columns=["userID", "itemID", "rating"])
    return Dataset.load_from_df(df, reader)


def train(db_path: str = DB_PATH, model_path: str = MODEL_PATH) -> SVD:
    data = _load_data(db_path)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(algo, f)
    print(f"Model trained and saved to {model_path}")
    return algo


def _get_model(model_path: str = MODEL_PATH) -> SVD:
    if not os.path.exists(model_path):
        return train(model_path=model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def top_k(user_id: int, k: int = 10, db_path: str = DB_PATH) -> list[int]:
    """Return top-k movie_ids predicted for user_id (unseen movies only)."""
    conn = get_db(db_path)
    seen = set(
        r[0] for r in conn.execute(
            "SELECT movie_id FROM ratings WHERE user_id=?", (user_id,)
        ).fetchall()
    )
    all_movies = set(
        r[0] for r in conn.execute("SELECT movie_id FROM movies").fetchall()
    )
    conn.close()

    algo = _get_model()
    candidates = all_movies - seen
    preds = [(mid, algo.predict(user_id, mid).est) for mid in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in preds[:k]]


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = top_k(1, k=10)
    print(f"top_k(user=1, k=10) = {result}")
    print(f"Time: {time.time() - t0:.2f}s")
