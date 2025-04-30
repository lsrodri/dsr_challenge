# persistence.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def save_results(df_scores: pd.DataFrame, params: dict, out_dir='experiments'):
    """
    - df_scores: DataFrame of CV results
    - params: dict of experiment settings (e.g. transformers, threshold, cv-folds,…)
    Saves:
      experiments/YYYYMMDD_HHMMSS_scores.csv
      experiments/YYYYMMDD_HHMMSS_params.json
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = Path(out_dir)
    folder.mkdir(exist_ok=True)
    df_scores.to_csv(folder/f'{ts}_scores.csv', index=False)
    with open(folder/f'{ts}_params.json', 'w') as f:
        json.dump(params, f, indent=2)
