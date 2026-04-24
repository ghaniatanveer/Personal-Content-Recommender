<p align="center">
  <strong>MyRecs</strong><br/>
  <em>Personal content recommender — movies, series, books, videos &amp; songs</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## Overview

**MyRecs** learns from **your** ratings file (`my_ratings.csv`) and recommends items using **three** complementary approaches:

| Approach | What it uses |
|----------|----------------|
| **Content-based** | TF–IDF on genres + short reviews, cosine similarity to a weighted taste profile |
| **Collaborative-style** | Biased matrix factorisation (SGD), augmented with synthetic “community” users so learning works with a single user |
| **Semantic** | [Sentence-Transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` on title + genres + review (GPU if available) |

The demo ships with a **synthetic dataset** (150 items). You can replace it with your own CSV using the same columns.

---

## Features

- Offline metrics: **Precision@5**, **Recall@5**, mean **rating of top-10** on a held-out split (`train.py` → `model/metrics.json`)
- **Streamlit** dashboard: tabbed recommendations with short explanations (see `app.py`; title text is customisable)
- **Self-contained Jupyter notebook** for [Google Colab](https://colab.research.google.com/): upload **only** `notebooks/recommender_notebook.ipynb` and **Runtime → Run all**
- Optional **technical report** with auto-generated figures: `docs/MyRecs_Project_Report.md` / `docs/generate_report_figures.py`

---

## Screenshots

*Evaluation figures (generated with `python docs/generate_report_figures.py`; commit `docs/screenshots/*.png` so they render on GitHub.)*

<p align="center">
  <img src="docs/screenshots/fig_notebook_metrics_bars.png" alt="Metrics comparison" width="720"/><br/>
  <sub>Offline metrics — TF-IDF vs MF vs embeddings</sub>
</p>

<p align="center">
  <img src="docs/screenshots/fig_notebook_rating_distribution.png" alt="Rating distribution" width="640"/><br/>
  <sub>Distribution of ratings in the synthetic catalogue</sub>
</p>

<p align="center">
  <img src="docs/screenshots/fig_notebook_tfidf_heatmap.png" alt="TF-IDF heatmap" width="640"/><br/>
  <sub>TF–IDF cosine similarity (sample of items)</sub>
</p>

> **Tip:** After cloning, run `python docs/generate_report_figures.py` once to create PNGs if they are missing, then commit them for README images.

---

## Example results

Hold-out evaluation (~20% of items). *Relevant* test items: rating **≥ 7**. Metrics computed on **held-out items only**.

| Method | Precision@5 | Recall@5 | Avg. rating (top-10, test pool) |
|--------|-------------|----------|-----------------------------------|
| TF–IDF | 0.40 | 0.125 | 7.90 |
| Biased MF | 0.40 | 0.125 | 6.40 |
| Sentence-Transformer | **0.60** | **0.188** | 6.20 |

*Numbers vary slightly if `data/my_ratings.csv` changes. Run `python train.py` to regenerate `model/metrics.json`.*

---

## Quick start (local)

**Requirements:** Python **3.10+**

```bash
git clone https://github.com/YOUR_USERNAME/MyRecs.git
cd MyRecs

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python train.py          # creates data if needed, prints metrics, saves model/svd_model.pkl
streamlit run app.py     # http://localhost:8501
```

---

## Google Colab (notebook only)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload **`notebooks/recommender_notebook.ipynb`** (single file — no repo zip required).
3. **Runtime → Run all** (first run downloads the embedding model ~90MB).

Session data is written under `/content/myrecs_data/` on Colab. Locally, the notebook uses `./myrecs_data/` when `/content` is not present.

---

## Project structure

```text
MyRecs/
├── app.py                 # Streamlit UI
├── train.py               # Train/evaluate all methods, save metrics + SVD pickle
├── requirements.txt
├── README.md
├── data/
│   └── my_ratings.csv     # 150 synthetic rows (generated if missing)
├── model/
│   ├── metrics.json       # from train.py
│   └── svd_model.pkl      # optional cache (gitignored)
├── src/
│   ├── data_preparation.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── embedding_recommender.py
│   └── utils.py
├── notebooks/
│   ├── recommender_notebook.ipynb   # standalone Colab/local notebook
│   └── _build_colab_standalone.py   # regenerates the notebook from templates
└── docs/
    ├── MyRecs_Project_Report.md     # full technical report
    ├── generate_report_figures.py   # PNG figures for docs/README
    ├── build_report_docx.py           # optional Word export
    └── screenshots/                 # figure PNGs (for README & report)
```

---

## Data format

`data/my_ratings.csv` columns:

| Column | Description |
|--------|-------------|
| `item_id` | Integer ID |
| `title` | Title string |
| `categories` | Comma-separated tags (e.g. `movie,sci-fi,drama`) |
| `my_rating` | Integer **1–10** |
| `short_review` | Short English review |

---

## Documentation

- **Technical report (PDF/Word):** edit `docs/MyRecs_Project_Report.md`, then run `python docs/build_report_docx.py` for `docs/MyRecs_Project_Report.docx`.
- **Regenerate README figures:** `python docs/generate_report_figures.py` from the repo root.

---

## Tech stack

- **Python:** pandas, NumPy, scikit-learn  
- **NLP:** sentence-transformers, PyTorch (optional CUDA)  
- **UI:** Streamlit  
- **Charts:** matplotlib, seaborn  

---

## Why not `scikit-surprise`?

This repo uses a **NumPy implementation** of biased matrix factorisation so installs work on **current Python** (e.g. 3.12+). The idea matches classic **SVD-style** collaborative filtering. If you use Python **3.10–3.11** and want the official library:

```bash
pip install scikit-surprise
```

---

## Roadmap / ideas

- More **real** ratings over time  
- **Hybrid** blending of TF–IDF, MF, and embedding scores  
- Hyperparameter tuning and cross-validation  
- Richer metadata (directors, channels, authors)

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) / [Hugging Face](https://huggingface.co/) for pre-trained models  
- scikit-learn for TF–IDF and utilities  

<p align="center">
  <sub>Built for coursework / portfolio — customise the app title and CSV for your own library.</sub>
</p>
