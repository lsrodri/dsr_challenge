from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        """
        Custom transformer to drop specified columns from a DataFrame.

        Parameters:
        columns_to_drop (list): List of column names to drop.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for this transformer).

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform method to drop specified columns.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame with specified columns dropped.
        """
        X = X.copy()
        if self.columns_to_drop:
            X.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
        return X
    
    