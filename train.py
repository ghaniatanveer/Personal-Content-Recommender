#!/usr/bin/env python3
"""
Train / evaluate all recommenders and persist metrics + SVD model.
Run from project root: python train.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import collaborative as collab
from src import content_based as cb
from src import data_preparation
from src import embedding_recommender as emb
from src.utils import (
    METRICS_JSON,
    RATINGS_CSV,
    avg_rating_top_n,
    ensure_dirs,
    load_ratings,
    precision_recall_at_k,
    save_metrics,
    train_test_split_items,
)

SVD_PATH = ROOT / "model" / "svd_model.pkl"


def ensure_dataset() -> pd.DataFrame:
    ensure_dirs()
    if not RATINGS_CSV.exists():
        print("Generating synthetic dataset (150 items)...")
        data_preparation.generate_dataset()
    return load_ratings()


def evaluate_methods(df: pd.DataFrame, relevant_threshold: float = 7.0) -> dict:
    """
    Hold out a test fraction; relevant test items are those with rating >= threshold.
    Rank test items only; compute Precision@5, Recall@5, avg rating of top-10.
    """
    train_df, test_df = train_test_split_items(df, test_fraction=0.2, random_state=42)
    test_ids = test_df["item_id"].astype(int).tolist()
    rel = set(test_df.loc[test_df["my_rating"] >= relevant_threshold, "item_id"].astype(int))
    id_to_rating = dict(zip(df["item_id"].astype(int), df["my_rating"].astype(float)))

    all_ids = df["item_id"].astype(int).tolist()

    metrics: dict = {"train_size": len(train_df), "test_size": len(test_df), "methods": {}}

    # --- Content-based ---
    ranked_c, _ = cb.rank_all_items_content(train_df, test_df, top_n=len(test_df))
    p5, r5 = precision_recall_at_k(ranked_c, rel, k=5)
    avg10 = avg_rating_top_n(ranked_c, id_to_rating, n=10)
    metrics["methods"]["content_tfidf"] = {
        "precision_at_5": round(p5, 4),
        "recall_at_5": round(r5, 4),
        "avg_rating_top_10": round(avg10, 4),
    }

    # --- SVD ---
    svd_model = collab.train_svd(train_df, all_item_ids=all_ids)
    ranked_s, _ = collab.rank_candidates_svd(svd_model, test_ids, raw_uid=0, top_n=len(test_ids))
    p5s, r5s = precision_recall_at_k(ranked_s, rel, k=5)
    avg10s = avg_rating_top_n(ranked_s, id_to_rating, n=10)
    metrics["methods"]["svd"] = {
        "precision_at_5": round(p5s, 4),
        "recall_at_5": round(r5s, 4),
        "avg_rating_top_10": round(avg10s, 4),
    }

    # --- Embeddings ---
    ranked_e, _ = emb.rank_all_embedding(
        train_df, test_df, model_name=emb.DEFAULT_MODEL_NAME, top_n=len(test_df)
    )
    p5e, r5e = precision_recall_at_k(ranked_e, rel, k=5)
    avg10e = avg_rating_top_n(ranked_e, id_to_rating, n=10)
    metrics["methods"]["sentence_transformer"] = {
        "precision_at_5": round(p5e, 4),
        "recall_at_5": round(r5e, 4),
        "avg_rating_top_10": round(avg10e, 4),
    }

    return metrics


def fit_full_and_save(df: pd.DataFrame) -> None:
    """Train SVD on full data for deployment; content/embedding are stateless given CSV."""
    ensure_dirs()
    all_ids = df["item_id"].astype(int).tolist()
    model = collab.train_svd(df, all_item_ids=all_ids)
    with open(SVD_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved SVD model to {SVD_PATH}")


def print_metrics_table(m: dict) -> None:
    print("\n=== Offline metrics (hold-out test items) ===")
    for name, vals in m.get("methods", {}).items():
        print(
            f"{name:22s}  P@5={vals['precision_at_5']:.4f}  "
            f"R@5={vals['recall_at_5']:.4f}  AvgTop10={vals['avg_rating_top_10']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MyRecs training and evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, do not save SVD")
    parser.add_argument("--no-save", action="store_true", help="Do not save SVD after eval")
    args = parser.parse_args()

    df = ensure_dataset()
    print(f"Loaded {len(df)} items from {RATINGS_CSV}")

    m = evaluate_methods(df)
    save_metrics(m)
    print_metrics_table(m)
    print(f"\nMetrics written to {METRICS_JSON}")

    if not args.eval_only and not args.no_save:
        fit_full_and_save(df)


if __name__ == "__main__":
    main()
