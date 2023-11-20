"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Feature_engineering.py
AUTOR: Clara Bureu
FECHA: 18/11/2023
"""

# Imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from utils import CappingTransformer
from utils import calculate_null_percentage
from utils import calculate_outlier_percentage


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self):
        """
        Reads a CSV file from the path specified in self.input_path and 
        returns a pandas DataFrame.

        Returns:
        pandas.DataFrame: A pandas DataFrame object containing the data from
        the CSV file.
        """
        pandas_df = pd.read_csv(self.input_path)
        
        return pandas_df

    def data_transformation(self, df):
        """
        Transform the input data into the desired output data.
        
        """
        numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        target = ['SalePrice']

        # Removing columns
        data_cleaned = df.copy()

        # Calculate null and outlier percentages
        outlier_percentages = calculate_outlier_percentage(
            data_cleaned[numeric_features])
        null_percentages = calculate_null_percentage(data_cleaned)

        # Selecting the 2 features with highest number of outliers and the 6
        # with highest number of missings
        features_to_drop = outlier_percentages[:2]['Column'].tolist() + null_percentages[:6]['Column'].tolist()  # noqa: E501
        data_cleaned.drop(columns=features_to_drop, inplace=True)

        # Removing na
        data_cleaned.dropna(inplace=True)

        # Re-defining numeric and categorical faetures
        numeric_features = data_cleaned.select_dtypes(
            include=['int64', 'float64']).columns.drop(['Id'])
        categorical_features = data_cleaned.select_dtypes(
            include=['object']).columns

        # Combine all the preprocessing steps into a single pipeline for numeric features # noqa: E501
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Step 1: Impute missing values # noqa: E501
            ('scaler', StandardScaler()),  # Step 2: Scale the data
            ('capper', CappingTransformer(threshold=1.5))  # Step 3: Apply the CappingTransformer # noqa: E501
        ])

        # Create a ColumnTransformer to apply the numeric_pipeline
        # to numeric features
        numeric_transformer = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric_features)
            ],
            remainder='passthrough'
        )

        # Combine all the steps into a single pipeline
        full_pipeline = Pipeline([
            ('feature_transform', numeric_transformer)
        ])

        # numeric_data = data_cleaned[numeric_features]

        # full_pipeline.fit(X=numeric_data)
        # numeric_data_transformed = full_pipeline.transform(X=numeric_data)
        # numeric_data_transformed_df = pd.DataFrame(
        #     numeric_data_transformed, columns=numeric_features)

        numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars',
                            'TotalBsmtSF', 'FullBath', 'YearRemodAdd',
                            'YearBuilt', 'Fireplaces', 'MasVnrArea', 'LotArea',
                            'SalePrice']
        categorical_features = ['ExterQual', 'CentralAir']
        metadata_features = ['Id']

        final_features = numeric_features + categorical_features + metadata_features  # noqa: E501

        full_pipeline.fit(X=data_cleaned[final_features])
        final_data = full_pipeline.transform(X=data_cleaned[final_features])
        final_data_df = pd.DataFrame(final_data)
        df_transformed = final_data_df.rename(
            columns=lambda x: x.replace('numeric__', '').replace('categoric__', '').replace('remainder__', ''))  # noqa: E501

        return df_transformed

    def write_prepared_data(self, transformed_dataframe):
        """
        Write the prepared data to CSV.
        """
        transformed_dataframe.to_csv(self.output_path, index=False)
       
        return None

    def run(self):
  
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path='..\\data\\train.csv',
                               output_path='..\\data\\transformed_dataframe.csv').run()  # noqa: E501
