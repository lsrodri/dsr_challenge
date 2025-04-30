
## imputation function

class CityBasedImputer:
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

    def fit(self, data):
        # Loop through each city (1 and 0)
        for city in data[self.city_column].unique():
            # For each city, compute the mean for each column (excluding the city column)
            city_data = data[data[self.city_column] == city]
            self.city_means[city] = city_data.mean()

    def transform(self, data):
        # Create a copy of the data to avoid modifying the original
        data_imputed = data.copy()
        

        # Loop through each city (1 and 0)
        for city in data[self.city_column].unique():
            # For each city, fill missing values with the computed mean of that city
            city_data = data_imputed[data_imputed[self.city_column] == city]
            
            for col in city_data.columns:
                if col != self.city_column:
                    # Impute missing values for each column in the city-specific data
                    city_data[col] = city_data[col].fillna(self.city_means[city][col])
            
            # Update the imputed data
            data_imputed.update(city_data)
        
        return data_imputed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)