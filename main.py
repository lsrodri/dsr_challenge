import sys
from pathlib import Path

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from transformers.transformers_l import (
    DropColumnsTransformer,
    CityBasedImputer,
    RollingAverageTransformer,
)
from transformers.interpolation_imputation import InterpolationImputer
from transformers.custom_transformers_mine import (
    CitySelector,
    CyclicTransformer,
    OutlierRemover,
    TempHumidityLagTransformer,
    LaggedNDVIRainInteractionTransformer,
    LaggedHeatHumidityStressIndexTransformer,
    MultiFeatureLagTransformer,
    RollingRainSumTransformer,
    WeatherNDVIAnomalyTransformer,
)
from transformers.interpolation_imputation import InterpolationImputer


from analysis.persistence import save_results

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
CONFIG = {
    "drop_cols1": [
        "week_start_date",
        "reanalysis_sat_precip_amt_mm",
        "reanalysis_dew_point_temp_k",
        "ndvi_nw",
        "ndvi_se",
        "reanalysis_air_temp_k",
        "reanalysis_tdtr_k",
    ],
    "drop_cols2": ["city"],
    "city_col": "city",
    "cycle_col": "weekofyear",
    "period": 52,
    "outlier_cfg": {
        "columns": [
            "ndvi_ne",
            "ndvi_sw",
            "precipitation_amt_mm",
            "reanalysis_avg_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_precip_amt_kg_per_m2",
            "reanalysis_relative_humidity_percent",
            "reanalysis_specific_humidity_g_per_kg",
            "station_avg_temp_c",
            "station_diur_temp_rng_c",
            "station_max_temp_c",
            "station_min_temp_c",
            "station_precip_mm",
        ],
        "remove": False,
        "method": "zscore",
        "threshold": 1.5,
    },
    "cv_folds": 5,
}


def build_pipeline(cfg):
    """
    Build a single Pipeline using RandomForestRegressor,
    """
    return Pipeline(
        [
            # 1) city-based imputation
            ("imputer_city", CityBasedImputer(city_column=cfg["city_col"])),
            # # Interaction features
            # (
            #     "temp_hum_lag",
            #     TempHumidityLagTransformer(
            #         tmin_col="reanalysis_min_air_temp_k",
            #         tmax_col="reanalysis_max_air_temp_k",
            #         rh_col="reanalysis_relative_humidity_percent",
            #         out_temp="temp_k",
            #         out_inter="temp_humidity",
            #         lags=(1, 3, 5),
            #         city_col=cfg["city_col"],
            #         sort_cols=["year", "weekofyear"],
            #     ),
            # ),
            # # if you want additional lags, you can chain another instance:
            # # ("temp_hum_lag3", TempHumidityLagTransformer(..., lag=3, ...)),
            # # 4) compute NDVI×rain and lag it by weeks 1,3,5
            # (
            #     "ndvi_rain_lags",
            #     LaggedNDVIRainInteractionTransformer(
            #         ndvi_cols=("ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"),
            #         rain_col="precipitation_amt_mm",
            #         out_ndvi="ndvi_mean",
            #         out_inter="rain_ndvi",
            #         lags=(1, 3, 5),
            #         city_col=cfg["city_col"],
            #         keep_ndvi_lags=False,
            #     ),
            # ),
            # (
            #     "heat_stress_lags",
            #     LaggedHeatHumidityStressIndexTransformer(
            #         lags=(1, 3, 5), city_col=cfg["city_col"], keep_range_lags=True
            #     ),
            # ),
            # (
            #     "lag_feats",
            #     MultiFeatureLagTransformer(
            #         feature_cols=[
            #             "precipitation_amt_mm",
            #             "ndvi_mean",
            #             "reanalysis_avg_temp_k",
            #         ],
            #         lags=[2, 4, 6],
            #         city_col="city",
            #         sort_cols=["year", "weekofyear"],
            #         fill_value=0,
            #     ),
            # ),
            # (
            #     "rolling_rain_4w",
            #     RollingRainSumTransformer(
            #         window=4,
            #         city_col=cfg["city_col"],
            #         rain_col="precipitation_amt_mm",
            #         out_col="rolling_rain_4w",
            #     ),
            # ),
            # (
            #     "anomalies",
            #     WeatherNDVIAnomalyTransformer(
            #         anomaly_cols=[
            #             "ndvi_mean",
            #             "precipitation_amt_mm",
            #             "reanalysis_avg_temp_k",
            #         ],
            #         date_col="week_start_date",  # or None if you already have a 'month' column
            #         month_col="month",  # where the transformer writes/expects your month
            #         group_cols=["city", "month"],  # grouping keys
            #         inplace=False,  # keep both raw & anomaly columns
            #     ),
            # ),
            # # 3) rolling-average features per city
            # (
            #     "rolling_avg",
            #     RollingAverageTransformer(
            #         columns=[
            #             "ndvi_ne",
            #             "ndvi_nw",
            #             "ndvi_se",
            #             "ndvi_sw",
            #             "precipitation_amt_mm",
            #             "reanalysis_air_temp_k",
            #             "reanalysis_avg_temp_k",
            #             "reanalysis_max_air_temp_k",
            #             "reanalysis_min_air_temp_k",
            #             "reanalysis_precip_amt_kg_per_m2",
            #             "reanalysis_relative_humidity_percent",
            #             "reanalysis_specific_humidity_g_per_kg",
            #             "reanalysis_tdtr_k",
            #             "station_avg_temp_c",
            #             "station_diur_temp_rng_c",
            #             "station_max_temp_c",
            #             "station_min_temp_c",
            #             "station_precip_mm",
            #         ],
            #         window=3,
            #         city_column=cfg["city_col"],
            #     ),
            # ),
            # 2) one-hot encode city
            ("city_sel", CitySelector(city=None)),
            # 4) drop raw columns
            ("drop1", DropColumnsTransformer(columns_to_drop=cfg["drop_cols1"])),
            ("drop2", DropColumnsTransformer(columns_to_drop=cfg["drop_cols2"])),
            # 5) cyclic encode the week
            (
                "cycle",
                CyclicTransformer(
                    column=cfg["cycle_col"],
                    period=cfg["period"],
                    drop_original=True,
                ),
            ),
            # 6) remove outliers
            #     ("outlier", OutlierRemover(**cfg["outlier_cfg"])),
            # 7) fill any remaining gaps & scale
            #     ("imputer_final", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            # 8) the Random Forest model
            ("estimator", RandomForestRegressor(n_estimators=100, random_state=0)),
        ]
    )


def main():
    # 1) Load & prep train
    feats = pd.read_csv("src/data/raw/dengue_features_train.csv")
    labels = pd.read_csv("src/data/raw/dengue_labels_train.csv")
    df = feats.merge(labels, on=["city", "year", "weekofyear"], how="inner")
    df["total_cases"] = np.log1p(df["total_cases"])

    X = df.drop(columns=["total_cases"])
    y = df["total_cases"]

    # 2) Build our single pipeline
    pipe = build_pipeline(CONFIG)

    # 3) TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=CONFIG["cv_folds"])
    scoring = ("r2", "neg_mean_squared_error", "neg_mean_absolute_error")

    print("⏱ Running time-aware CV…")
    cvres = cross_validate(
        pipe,
        X,
        y,
        cv=tscv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # aggregate
    results = {"metric": [], "train_mean": [], "val_mean": []}
    for metric in scoring:
        tr = cvres[f"train_{metric}"].mean()
        te = cvres[f"test_{metric}"].mean()
        name = metric[4:] if metric.startswith("neg_") else metric
        if metric.startswith("neg_"):
            tr, te = -tr, -te
        results["metric"].append(name)
        results["train_mean"].append(tr)
        results["val_mean"].append(te)

    df_scores = pd.DataFrame(results)
    print("\n=== CV RESULTS ===")
    print(df_scores)

    save_results(df_scores, {"config": CONFIG, "model": "RandomForest"})

    # 4) Retrain on full data & predict test
    print("\n🏋️ Retraining on full training set…")
    pipe.fit(X, y)

    test = pd.read_csv("src/data/raw/dengue_features_test.csv")
    submission = test[["city", "year", "weekofyear"]].copy()

    # undo log1p
    preds_log = pipe.predict(test)
    preds = np.expm1(preds_log).round().astype(int)
    submission["total_cases"] = preds

    # 5) Write
    out_path = "src/data/predictions/one_model_random_forest_simple.csv"
    submission.to_csv(out_path, index=False)
    print(f"✅ Wrote submission to {out_path}")


if __name__ == "__main__":
    main()
