"""Fit and persist the movie embedding pipeline.

Reproduces aaastreamer_3/movie_embeddings.ipynb, then fits PCA to exactly
N_COMPONENTS (390 -> ~90% explained variance) and saves every fitted object
needed to embed *new* movies in the same space.

Usage:
    python -m scripts.fit_embeddings [path/to/movies.csv]
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from app.ml import config, features


def df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "plot": row.get("plot"),
            "genre": row.get("genre"),
            "director": row.get("director"),
            "writer": row.get("writer"),
            "actors": row.get("actors"),
            "language": row.get("language"),
            "country": row.get("country"),
            "year": row.get("year"),
        }
        for row in df.to_dict(orient="records")
    ]


def main(csv_path: str) -> None:
    t0 = time.time()
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} movies")

    # --- fit the stateful transformers ---------------------------------- #
    genre_lists = [features.clean_tokens(x) for x in df["genre"]]
    mlb = MultiLabelBinarizer().fit(genre_lists)
    print(f"  genre classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

    years = df["year"].astype(float).values.reshape(-1, 1)
    median_year = float(np.nanmedian(years))
    years_filled = np.where(np.isnan(years), median_year, years).astype(np.float32)
    scaler = StandardScaler().fit(years_filled)
    print(f"  median year (fill): {median_year:.0f}")

    # --- build the 732-d feature matrix --------------------------------- #
    print("Loading MiniLM and encoding plots (this is the slow part) ...")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(config.MODEL_NAME)
    records = df_to_records(df)
    matrix = features.build_feature_matrix(
        records, st_model, mlb, scaler, median_year
    )
    print(f"  feature matrix: {matrix.shape}")

    # --- fit PCA to the schema width (390) ------------------------------ #
    print(f"Fitting PCA -> {config.N_COMPONENTS} components ...")
    pca = PCA(
        n_components=config.N_COMPONENTS, svd_solver="full", random_state=42
    ).fit(matrix)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  explained variance @ {config.N_COMPONENTS} dims: {explained:.4f}")

    # --- persist -------------------------------------------------------- #
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "genre_mlb": mlb,
            "year_scaler": scaler,
            "median_year": median_year,
            "pca": pca,
            "model_name": config.MODEL_NAME,
            "explained_variance": explained,
            "weights": config.WEIGHTS,
            "n_components": config.N_COMPONENTS,
        },
        config.PIPELINE_PATH,
    )
    print(f"Saved pipeline -> {config.PIPELINE_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(config.DEFAULT_MOVIES_CSV)
    main(path)
