from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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

class CityBasedImputer(BaseEstimator, TransformerMixin):
    '''
    Impute missing column values with mean, differentiated by city.

    Requirements:
        - column "start_week_date" is dropped
        - column "city" is encoded into numerical values

    To apply call:
    imputer = CityBasedImputer(city_column='city')
    df_imputed = imputer.fit_transform(df)
    '''
    def __init__(self, city_column):
        self.city_column = city_column
        self.city_means = {}

    def fit(self, X, y=None):  # Changed parameter name and added y=None
        """
        Fit method to calculate city-specific means for each column.
        
        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).
        
        Returns:
        self: Returns the transformer instance.
        """
        
        #check if X['city'] has string values
        if X[self.city_column].dtype == 'object':
            # Encode the city column to numerical values
            X['city'] = X[self.city_column].map({'sj': 0, 'iq': 1})

        # Loop through each city
        for city in X[self.city_column].unique():
            # For each city, compute the mean for each column
            city_data = X[X[self.city_column] == city]
            self.city_means[city] = city_data.mean(numeric_only=True)
        
        return self  # Return self for method chaining

    def transform(self, X):  # Changed parameter name
        """
        Transform method to impute missing values based on city-specific means.
        
        Parameters:
        X (pd.DataFrame): Input DataFrame.
        
        Returns:
        pd.DataFrame: Transformed DataFrame with missing values imputed.
        """
        # Create a copy of the data to avoid modifying the original
        X_imputed = X.copy()
        
        # Loop through each city
        for city in X[self.city_column].unique():
            if city in self.city_means:  # Check if city exists in city_means
                # For each city, fill missing values with the computed mean of that city
                city_mask = X_imputed[self.city_column] == city
                
                for col in self.city_means[city].index:
                    if col in X_imputed.columns:
                        X_imputed.loc[city_mask, col] = X_imputed.loc[city_mask, col].fillna(self.city_means[city][col])
        
        return X_imputed

    def fit_transform(self, X, y=None):  # Changed parameter names and added y=None
        """
        Fit and transform method to handle pipelines.
        
        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).
        
        Returns:
        pd.DataFrame: Transformed DataFrame with missing values imputed.
        """
        return self.fit(X, y).transform(X)