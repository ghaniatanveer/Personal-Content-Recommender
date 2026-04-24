"""
MyRecs — Streamlit dashboard: ratings, recommendations (TF-IDF, SVD, embeddings).
Run from project root: streamlit run app.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import collaborative as collab
from src import content_based as cb
from src import data_preparation
from src import embedding_recommender as emb
from src.utils import RATINGS_CSV, load_metrics, load_ratings

SVD_PATH = ROOT / "model" / "svd_model.pkl"


def _data_mtime() -> float:
    if not RATINGS_CSV.exists():
        return 0.0
    return RATINGS_CSV.stat().st_mtime


@st.cache_data(show_spinner=False)
def cached_ratings() -> pd.DataFrame:
    if not RATINGS_CSV.exists():
        data_preparation.generate_dataset()
    return load_ratings()


@st.cache_data(show_spinner="Scoring with TF-IDF…")
def cached_tfidf_rank(_mtime: float) -> tuple[list[int], dict[int, float]]:
    df = load_ratings()
    return cb.rank_full_catalog_tfidf(df, top_n=10)


@st.cache_data(show_spinner="Computing embeddings (first run downloads the model)…")
def cached_embedding_rank(_mtime: float) -> tuple[list[int], dict[int, float]]:
    df = load_ratings()
    return emb.rank_full_catalog_embedding(df, top_n=10)


@st.cache_resource(show_spinner="Loading collaborative model…")
def cached_svd_model(_mtime: float):
    df = load_ratings()
    all_ids = df["item_id"].astype(int).tolist()
    if SVD_PATH.exists():
        with open(SVD_PATH, "rb") as f:
            return pickle.load(f)
    model = collab.train_svd(df, all_item_ids=all_ids)
    SVD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SVD_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def row_by_id(df: pd.DataFrame, item_id: int) -> pd.Series:
    return df.loc[df["item_id"] == item_id].iloc[0]


def main() -> None:
    st.set_page_config(
        page_title="MyRecs — Recommendations for Haseeb",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.header("My Ratings")
    df = cached_ratings()
    st.sidebar.dataframe(
        df[["item_id", "title", "my_rating"]].sort_values("my_rating", ascending=False),
        use_container_width=True,
        height=320,
    )
    st.sidebar.caption(f"{len(df)} items in `data/my_ratings.csv`")

    st.title("Recommendations for Haseeb")
    st.caption("Personal Content Recommender — movies, series, books, YouTube, and songs.")

    metrics = load_metrics()
    if metrics:
        with st.expander("Latest offline metrics (from `python train.py`)"):
            st.json(metrics.get("methods", {}))

    tab1, tab2, tab3 = st.tabs(["Content-based (TF-IDF)", "Collaborative (SVD)", "Embeddings (MiniLM)"])

    mt = _data_mtime()
    ranked_tfidf, scores_tfidf = cached_tfidf_rank(mt)
    svd_model = cached_svd_model(mt)
    ranked_svd, scores_svd = collab.rank_candidates_svd(
        svd_model, df["item_id"].astype(int).tolist(), raw_uid=0, top_n=10
    )
    ranked_emb, scores_emb = cached_embedding_rank(mt)

    def show_tab(ranked_ids: list[int], score_map: dict[int, float], kind: str) -> None:
        rows = []
        for iid in ranked_ids:
            r = row_by_id(df, iid)
            sc = score_map.get(iid, 0.0)
            if kind == "tfidf":
                expl = cb.explain_content_based(df, r, sc)
                score_label = f"Similarity: {sc:.3f}"
            elif kind == "svd":
                expl = collab.explain_svd(sc, df)
                score_label = f"Predicted rating: {sc:.2f}/10"
            else:
                expl = emb.explain_embedding(sc, df)
                score_label = f"Embedding sim: {sc:.3f}"

            rows.append(
                {
                    "Title": r["title"],
                    "Score": score_label,
                    "Your rating": f"{int(r['my_rating'])}/10",
                    "Genres / tags": r["categories"],
                    "Why": expl,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab1:
        st.subheader("TF-IDF + cosine similarity")
        show_tab(ranked_tfidf, scores_tfidf, "tfidf")

    with tab2:
        st.subheader("Collaborative — biased MF / SGD (synthetic community users)")
        show_tab(ranked_svd, scores_svd, "svd")

    with tab3:
        st.subheader("Sentence-Transformers (all-MiniLM-L6-v2)")
        device = emb.get_device()
        st.caption(f"Encoder device: **{device}**")
        show_tab(ranked_emb, scores_emb, "emb")

    st.divider()
    st.subheader("Add a new item (updates dataset & retrains SVD)")
    with st.form("add_item"):
        c1, c2 = st.columns(2)
        title = c1.text_input("Title")
        cats = c2.text_input("Categories (comma-separated)", "movie,drama,feel-good")
        rating = st.slider("Your rating", 1, 10, 7)
        review = st.text_area("Short review", "Honestly better than I expected.")
        submitted = st.form_submit_button("Save & retrain")
    if submitted:
        if not title.strip():
            st.error("Title is required.")
        else:
            new_id = int(df["item_id"].max()) + 1
            new_row = pd.DataFrame(
                [
                    {
                        "item_id": new_id,
                        "title": title.strip(),
                        "categories": cats.strip(),
                        "my_rating": rating,
                        "short_review": review.strip() or "No review yet.",
                    }
                ]
            )
            out = pd.concat([df, new_row], ignore_index=True)
            RATINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(RATINGS_CSV, index=False)
            if SVD_PATH.exists():
                SVD_PATH.unlink()
            cached_ratings.clear()
            cached_tfidf_rank.clear()
            cached_embedding_rank.clear()
            cached_svd_model.clear()
            st.success(f"Saved item {new_id}. Reloading…")
            st.rerun()


if __name__ == "__main__":
    main()
