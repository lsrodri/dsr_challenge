# main.py

import multiprocessing as mp

# Use forkserver instead of fork
mp.set_start_method('forkserver', force=True)


import sys
from pathlib import Path

# make sure src/ is on the import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer           # ← add this import


# your custom transformers
from transformers.transformers_l import DropColumnsTransformer, CityBasedImputer
from transformers.custom_transformers_mine import CitySelector, CyclicTransformer, OutlierRemover

# models, evaluation, and persistence
from models.models       import get_models
from analysis.evaluation import evaluate_pipelines
from analysis.persistence import save_results


def build_pipelines(models, config):
    """
    Assemble a pipeline for each model, in the order:
      1) City-based imputation
      2) CitySelector (one-hot when city=None)
      3) DropColumnsTransformer
      4) CyclicTransformer
      5) OutlierRemover
      6) StandardScaler
      7) Estimator
    """
    pipes = {}
    for name, mdl in models.items():
        pipes[name] = Pipeline([
            ('drop_cols1',    DropColumnsTransformer(columns_to_drop=config['drop_cols1'])),
            ('imputer_city', CityBasedImputer(city_column=config['city_col'])),
            ('city_sel',     CitySelector(city=None)),
            ('drop_cols2',    DropColumnsTransformer(columns_to_drop=config['drop_cols2'])),
            ('cyclic',       CyclicTransformer(
                                 column=config['cycle_col'],
                                 period=config['period'],
                                 drop_original=True
                             )),
            ('outlier',      OutlierRemover(**config['outlier_cfg'])),
            ('imputer_final', SimpleImputer(strategy='mean')),   # <<< fills any remaining NaNs
            ('scaler',       StandardScaler()),
            ('estimator',    mdl)
        ])
    return pipes


def main():
    # 1) Load raw feature and label data
    data   = pd.read_csv("src/data/raw/dengue_features_train.csv")
    target = pd.read_csv("src/data/raw/dengue_labels_train.csv")

    # merge on the key columns
    df = pd.merge(data, target, on=['city', 'year', 'weekofyear'], how='inner')

    # 2) Log-transform the target
    df['total_cases'] = np.log1p(df['total_cases'])

    # split features / target
    X = df.drop(columns=['total_cases'])
    y = df['total_cases']

    # 3) Pipeline configuration
    config = {
    # correct spelling ↓↓↓
    'drop_cols1':   ['week_start_date', 
                     'reanalysis_sat_prescip_amt_mm',
                     'reanalysis_dew_point_temp_k'],  
    'drop_cols2':   ['city'],                   # drop the date string column
    'city_col':    'city',
    'cycle_col':   'weekofyear',
    'period':      52,
    'outlier_cfg': {'columns': None, 'method': 'zscore', 'threshold': 3.0}
}

    # 4) Instantiate models and pipelines
    models    = get_models(random_state=0)
    pipelines = build_pipelines(models, config)

    # 5) Evaluate with cross-validation
    df_scores = evaluate_pipelines(pipelines, X, y, cv=5)
    print(df_scores)

    # 6) Save experiment details & results
    params = {
        'drop_cols':  config['drop_cols'],
        'city_col':   config['city_col'],
        'cycle_cfg':  {'column': config['cycle_col'], 'period': config['period']},
        'outlier_cfg': config['outlier_cfg'],
        'models':     list(models.keys()),
        'cv_folds':   5
    }
    save_results(df_scores, params)


if __name__ == "__main__":
    main()
