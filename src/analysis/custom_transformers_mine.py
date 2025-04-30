from sklearn.base import BaseEstimator, TransformerMixin

class CitySelector(BaseEstimator, TransformerMixin):
    """
    A custom transformer that selects rows for a given city, or returns all rows if city=None.
    
    Parameters
    ----------
    city : str or None, default=None
        - If a string like 'sj' or 'iq', only rows where df['city'] == city are returned.
        - If None, the full DataFrame is passed through.
    """
    def __init__(self, city=None):
        self.city = city

    def fit(self, X, y=None):
        # No fitting necessary
        return self

    def transform(self, X):
        # Expect X to be a pandas DataFrame
        if self.city is None:
            return X.copy()
        # Filter and reset index so downstream transformers see a clean index
        return X[X['city'] == self.city].reset_index(drop=True)
    



    