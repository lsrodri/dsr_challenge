# main.py

import sys
from pathlib import Path

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from transformers.transformers_l import DropColumnsTransformer, CityBasedImputer
from transformers.interpolation_imputation import InterpolationImputer
from transformers.custom_transformers_mine import (
    CitySelector,
    CyclicTransformer,
    OutlierRemover,
)

from models.models import get_models
from analysis.evaluation import evaluate_pipelines, evaluate_pipelines_timeaware
from analysis.persistence import save_results


CONFIG = {
    "drop_cols1": [
        "week_start_date",
        "reanalysis_sat_precip_amt_mm",
        "reanalysis_dew_point_temp_k",
    ],
    "drop_cols2": ["city"],
    "city_col": "city",
    "cycle_col": "weekofyear",
    "period": 52,
    "outlier_cfg": {
        "columns": [
            "ndvi_ne",
            "ndvi_nw",
            "ndvi_se",
            "ndvi_sw",
            "precipitation_amt_mm",
            "reanalysis_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_precip_amt_kg_per_m2",
            "reanalysis_relative_humidity_percent",
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_tdtr_k",
            "station_avg_temp_c",
            "station_diur_temp_rng_c",
            "station_max_temp_c",
            "station_min_temp_c",
            "station_precip_mm",
        ],
        "remove": False,
        "method": "zscore",
        "threshold": 3.0,
    },
    "cv_folds": 5,
    # ←── new keys for time-aware splitting ──→
    "ts_test_size": 20,  # how many samples in each test fold
    "ts_max_train_size": None,  # or an int to cap the training window
}


def build_pipelines(models, cfg):
    pipes = {}
    for name, mdl in models.items():
        pipes[name] = Pipeline(
            [
                ("imputer_city", CityBasedImputer(city_column=cfg["city_col"])),
                # ("imputer_city", InterpolationImputer()),
                ("city_sel", CitySelector(city=None)),
                ("drop1", DropColumnsTransformer(columns_to_drop=cfg["drop_cols1"])),
                ("drop2", DropColumnsTransformer(columns_to_drop=cfg["drop_cols2"])),
                (
                    "cycle",
                    CyclicTransformer(
                        column=cfg["cycle_col"],
                        period=cfg["period"],
                        drop_original=True,
                    ),
                ),
                ("outlier", OutlierRemover(**cfg["outlier_cfg"])),
                ("imputer_final", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("estimator", mdl),
            ]
        )
    return pipes


def run_city_experiment(city_code, df_train, df_test, models, config):
    """
    1) CV on df_train, select best by val_r2
    2) Retrain best on full df_train
    3) Predict on df_test
    4) Return a DataFrame with city,year,weekofyear,total_cases
    """
    X = df_train.drop(columns=["total_cases"])
    y = df_train["total_cases"]

    pipelines = build_pipelines(models, config)

    #    scores = evaluate_pipelines(pipelines, X, y, cv=config["cv_folds"])
    # 3) time-aware CV
    print("Running time-aware cross-validation…")
    scores = evaluate_pipelines_timeaware(
        pipelines,
        X,
        y,
        n_splits=CONFIG["cv_folds"],
        test_size=CONFIG["ts_test_size"],
        max_train_size=CONFIG["ts_max_train_size"],
        scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error"),
    )

    print(f"\n=== CV RESULTS for city={city_code} ===")
    print(scores)

    best_model = scores.sort_values("train_r2", ascending=False).iloc[0]["model"]
    print(f"--> Best model for {city_code}: {best_model}")

    best_pipe = pipelines[best_model]
    best_pipe.fit(X, y)

    # get predictions (log scale) and invert
    preds_log = best_pipe.predict(df_test)
    preds = np.expm1(preds_log).round().astype(int)

    sub = df_test[["city", "year", "weekofyear"]].copy()
    sub["total_cases"] = preds
    return sub


def main():
    # ────────────────────────────────────────────────────────
    # 1) Load & prep train
    # ────────────────────────────────────────────────────────
    feats = pd.read_csv("src/data/raw/dengue_features_train.csv")
    labels = pd.read_csv("src/data/raw/dengue_labels_train.csv")
    df = feats.merge(labels, on=["city", "year", "weekofyear"], how="inner")
    df["total_cases"] = np.log1p(df["total_cases"])

    X = df.drop(columns=["total_cases"])
    y = df["total_cases"]

    # ────────────────────────────────────────────────────────
    # 2) Build & CV all pipelines
    # ────────────────────────────────────────────────────────
    models = get_models(random_state=0)
    pipelines = build_pipelines(models, CONFIG)

    print("Running cross‐validation...")
    scores = evaluate_pipelines(
        pipelines,
        X,
        y,
        cv=CONFIG["cv_folds"],
        scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error"),
    )
    print("\n=== CV RESULTS ===")
    print(scores)

    # ────────────────────────────────────────────────────────
    # 3) Pick best by train_r2
    # ────────────────────────────────────────────────────────
    best_model = scores.sort_values("train_mean_absolute_error", ascending=False).iloc[
        0
    ]["model"]
    print(f"\n✅ Best model by TRAIN R²: {best_model}")

    # persist experiment metadata + scores
    save_results(
        scores,
        {"config": CONFIG, "models": list(models.keys()), "best_model": best_model},
    )

    # ────────────────────────────────────────────────────────
    # 4) Retrain on full train set
    # ────────────────────────────────────────────────────────
    best_pipe = pipelines[best_model]
    best_pipe.fit(X, y)

    # ────────────────────────────────────────────────────────
    # 5) Load test, predict, undo log
    # ────────────────────────────────────────────────────────
    test = pd.read_csv("src/data/raw/dengue_features_test.csv")
    submission = test[["city", "year", "weekofyear"]].copy()

    preds_log = best_pipe.predict(test)
    preds = np.expm1(preds_log).round().astype(int)
    submission["total_cases"] = preds

    # ────────────────────────────────────────────────────────
    # 6) Write submission.csv
    # ────────────────────────────────────────────────────────
    submission.to_csv(
        "src/data/predictions/one_model_interpolation_vik_1st_submission.csv",
        index=False,
    )
    print("✅ Wrote submission.csv")


if __name__ == "__main__":
    main()
