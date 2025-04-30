# main.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

# add the src directory to sys.path so Python can import packages from it
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from transformers.transformers_l import (
    DropColumnsTransformer,
    CityBasedImputer
)

from transformers.custom_transformers_mine import (
    CitySelector,
    CyclicTransformer,
    OutlierRemover
)

from models.models import get_models
from analysis.evaluation import evaluate_pipelines
from analysis.persistence import save_results

def build_pipelines(models, config):
    """
    Build a dict of pipelines for each model.

    config keys:
      - drop_cols: list of column names to drop
      - city_col:  name of the city column
      - cycle_col: name of the cyclical column (e.g. 'weekofyear')
      - period:    period for the cyclical transformer (e.g. 52)
      - outlier_cfg: dict to unpack into OutlierRemover
    """
    pipes = {}
    for name, mdl in models.items():
        pipes[name] = Pipeline([
            ('city_sel', CitySelector(city=None)),
            ('col_drop', DropColumnsTransformer(columns_to_drop=config['drop_cols'])),
            ('imputer',  CityBasedImputer(city_column=config['city_col'])),
            ('cyclic',   CyclicTransformer(
                            column=config['cycle_col'],
                            period=config['period'],
                            drop_original=True
                        )),
            ('outlier',  OutlierRemover(**config['outlier_cfg'])),
            ('scaler',   StandardScaler()),
            ('estimator', mdl)
        ])
    return pipes

def main():
    # 1. Load data
    data=pd.read_csv("src/data/raw/dengue_features_train.csv") # Train data
    target=pd.read_csv("src/data/raw/dengue_labels_train.csv") # Labels

    # Merge data
    df = pd.merge(data, target, on=['city', 'year', 'weekofyear'], how='inner')


    # 2. Log-transform the target
    df['total_cases'] = np.log1p(df['total_cases'])

    X = df.drop(columns=['total_cases'])
    y = df['total_cases']

    # 3. Configuration
    config = {
        'drop_cols':    ['start_week_date'],
        'city_col':     None,
        'cycle_col':    'weekofyear',
        'period':       52,
        'outlier_cfg':  {'columns': None, 'method': 'zscore', 'threshold': 3.0}
    }

    # 4. Build and evaluate
    models = get_models(random_state=0)
    pipelines = build_pipelines(models, config)

    df_scores = evaluate_pipelines(pipelines, X, y, cv=5)
    print(df_scores)

    # 5. Save experiment metadata + results
    params = {
        'drop_cols':    config['drop_cols'],
        'city_col':     config['city_col'],
        'cycle_cfg':    {'column': config['cycle_col'], 'period': config['period']},
        'outlier_cfg':  config['outlier_cfg'],
        'models':       list(models.keys()),
        'cv_folds':     5
    }
    save_results(df_scores, params)

if __name__ == '__main__':
    main()
