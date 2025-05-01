# main.py

import sys
from pathlib import Path

# ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from transformers.transformers_l import DropColumnsTransformer, CityBasedImputer
from transformers.custom_transformers_mine import (
    CitySelector,
    CyclicTransformer,
    OutlierRemover,
)

from models.models import get_models
from analysis.evaluation import evaluate_pipelines, evaluate_pipelines_timeaware
from analysis.persistence import save_results


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

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


def build_pipelines(models, config):
    pipes = {}
    for name, mdl in models.items():
        pipes[name] = Pipeline(
            [
                ("imputer_city", CityBasedImputer(city_column=config["city_col"])),
                ("city_sel", CitySelector(city=None)),
                ("drop1", DropColumnsTransformer(columns_to_drop=config["drop_cols1"])),
                ("drop2", DropColumnsTransformer(columns_to_drop=config["drop_cols2"])),
                (
                    "cycle",
                    CyclicTransformer(
                        column=config["cycle_col"],
                        period=config["period"],
                        drop_original=True,
                    ),
                ),
                ("outlier", OutlierRemover(**config["outlier_cfg"])),
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
    # 1) Load & merge train
    feats = pd.read_csv("src/data/raw/dengue_features_train.csv")
    labels = pd.read_csv("src/data/raw/dengue_labels_train.csv")
    df = feats.merge(labels, on=["city", "year", "weekofyear"], how="inner")
    df["total_cases"] = np.log1p(df["total_cases"])

    # 2) Load raw test (for keys & pipeline input)
    test = pd.read_csv("src/data/raw/dengue_features_test.csv")

    # 3) Prepare models
    models = get_models(random_state=0)

    # 4) Run experiment per city
    submissions = []
    for city_code in df["city"].unique():
        df_tr = df[df["city"] == city_code].reset_index(drop=True)
        df_te = test[test["city"] == city_code].reset_index(drop=True)

        sub_city = run_city_experiment(city_code, df_tr, df_te, models, CONFIG)
        submissions.append(sub_city)

    # 5) Build final submission in original test order
    submission = test[["city", "year", "weekofyear"]].copy()
    all_preds = pd.concat(submissions, axis=0)
    lookup = {
        (r.city, r.year, r.weekofyear): r.total_cases for _, r in all_preds.iterrows()
    }
    submission["total_cases"] = submission.apply(
        lambda r: lookup[(r.city, r.year, r.weekofyear)], axis=1
    )

    # 6) Save
    submission.to_csv(
        "src/data/predictions/two_models_time_aware_CV_best_train_r2.csv", index=False
    )
    print("Wrote submission.csv in correct test‐file order.")


if __name__ == "__main__":
    main()
