import sys
from pathlib import Path

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent / "../src"))

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


from transformers.transformers_l import (
    DropColumnsTransformer, 
    CityBasedImputer, 
    RollingAverageTransformer,
    CityMapTransformer)

from transformers.interpolation_imputation import InterpolationImputer
from transformers.custom_transformers_mine import (
    CitySelector,
    CyclicTransformer,
    OutlierRemover,
)

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

moving_features = [
    'ndvi_ne',
    'ndvi_sw',
    'precipitation_amt_mm',
    'reanalysis_avg_temp_k',
    'reanalysis_max_air_temp_k',
    'reanalysis_min_air_temp_k',
    # 'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent',
    # 'reanalysis_specific_humidity_g_per_kg',
    'station_avg_temp_c',
    'station_diur_temp_rng_c',
    'station_max_temp_c',
    # 'station_min_temp_c',
    'station_precip_mm'
]

second_drop_columns = [
    'reanalysis_precip_amt_kg_per_m2',
    'station_min_temp_c',
 ]

def build_pipeline(cfg):
    """
    Build a single Pipeline using RandomForestRegressor.
    """
    return Pipeline(
        [
            ('drop_columns', DropColumnsTransformer(columns_to_drop=[
                     'week_start_date', 
                     'reanalysis_sat_prescip_amt_mm',
                     'reanalysis_dew_point_temp_k',
                     'ndvi_nw',
                     'ndvi_se',
                     'reanalysis_air_temp_k',
                     'reanalysis_tdtr_k'])),
                    ('city_encoder', CityMapTransformer()),
                    ('imputer', CityBasedImputer(city_column='city')),
                    ('rolling_avg', RollingAverageTransformer(
                        columns=moving_features, 
                        window=3)),
                    ('drop_rolling_columns', DropColumnsTransformer(columns_to_drop=moving_features)),
                    ('drop_extra_columns', DropColumnsTransformer(columns_to_drop=second_drop_columns)),
                    ('city_onehot', ColumnTransformer([('onehot', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), ['city'])], remainder='passthrough')), 
                    ('final_imputer', SimpleImputer(strategy='mean')),
                    ('model', RandomForestRegressor(
                        n_estimators=100,
                        random_state=0))
        ]
    )

def optimize_pipeline(X, y, cv_splitter):
    # Create a conversion function to preserve DataFrame structure
    class ArrayToDataFrame(BaseEstimator, TransformerMixin):
        def __init__(self, columns=None):
            self.columns = columns
            
        def fit(self, X, y=None):
            return self
            
        def transform(self, X):
            if not isinstance(X, pd.DataFrame):
                # If columns were not provided, use original columns from X
                cols = self.columns if self.columns is not None else X.columns if hasattr(X, 'columns') else None
                return pd.DataFrame(X, columns=cols)
            return X
    
    
    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('drop_columns', DropColumnsTransformer(columns_to_drop=[
                        'week_start_date', 
                        'reanalysis_sat_prescip_amt_mm',
                        'reanalysis_dew_point_temp_k',
                        'ndvi_nw',
                        'ndvi_se',
                        'reanalysis_air_temp_k',
                        'reanalysis_tdtr_k'])),
        ('city_encoder', CityMapTransformer()),
        ('imputer', CityBasedImputer(city_column='city')),
        ('rolling_avg', RollingAverageTransformer(
            columns=moving_features, 
            window=3)),
        ('drop_rolling_columns', DropColumnsTransformer(columns_to_drop=moving_features)),
        ('drop_extra_columns', DropColumnsTransformer(columns_to_drop=second_drop_columns)),
        ('city_onehot', ColumnTransformer([
        ('onehot', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), ['city'])], remainder='passthrough')), 
        # Add these two steps to handle NaN and extreme values
        ('final_imputer', SimpleImputer(strategy='mean')),
        ('to_df', ArrayToDataFrame()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            random_state=0))
    ])

    # Define parameter grid
    param_grid = {
        # Model complexity parameters - add 'model__' prefix
        'model__n_estimators': [50, 100, 200, 300, 500],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_leaf': [1, 3, 5],
        
        # Feature engineering parameters
        'rolling_avg__window': [2, 3, 5, 7],
        
        # Control whether to keep or drop original features
        'drop_rolling_columns__columns_to_drop': [
            [],  # Keep all original features
            moving_features  # Drop original features, keep only rolling averages
        ]
    }
    
    # Set up grid search with time series CV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splitter,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2
    )
    
    # Run grid search
    grid_search.fit(X, y)
    
    # Print best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best MAE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.cv_results_

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
    out_path = "src/data/predictions/one_model_random_forest_rolling_week_3_drop_More_columns.csv"
    submission.to_csv(out_path, index=False)
    print(f"✅ Wrote submission to {out_path}")

    # Set up time series CV
    tscv = TimeSeriesSplit(n_splits=CONFIG["cv_folds"])
    
    # Run optimization
    best_model, cv_results = optimize_pipeline(X, y, tscv)
    
    # Create results dataframe for detailed analysis
    results_df = pd.DataFrame(cv_results)
    
    # Save detailed results
    results_df.to_csv("grid_search_results.csv")
    
    # Use best model for final prediction
    test = pd.read_csv("src/data/raw/dengue_features_test.csv")
    submission = test[["city", "year", "weekofyear"]].copy()
    preds_log = best_model.predict(test)
    preds = np.expm1(preds_log).round().astype(int)
    submission["total_cases"] = preds
    
    # Write submission
    out_path = "src/data/predictions/optimized_model.csv"
    submission.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
