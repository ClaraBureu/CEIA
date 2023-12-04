"""
feature_engineering.py

This script defines a FeatureEngineeringPipeline class to perform data preparation tasks.

Imports:
- os
- pandas as pd
- Pipeline, SimpleImputer, StandardScaler, ColumnTransformer, OneHotEncoder from sklearn
- CappingTransformer from utils
- train_test_split from sklearn.model_selection
- argparse for command-line arguments

Classes:
- FeatureEngineeringPipeline: A class for data preparation, including methods for reading data,
  transforming it based on modes, writing transformed data to CSV, and executing
  the entire pipeline.

Usage:
- Execute this script with 'train' or 'test' as the argument to prepare training or test data,
respectively.
- Ensure the necessary input files ('train.csv' for 'train' mode or 'test.csv' for 'test' mode)
exist in the 'data' directory.

DESCRIPCIÓN: feature_engineering.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

# Imports
import argparse
import logging
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from utils import CappingTransformer


# Log setting
logging.basicConfig(level=logging.DEBUG,
                    filename='data_logger.log',
                    filemode='a',
                    format='%(asctime)s:%(levelname)s:%(message)s')


class FeatureEngineeringPipeline(object):
    """
    A class for data preparation pipeline.

    This class provides methods for reading, transforming, and writing data for machine
    learning tasks.

    Parameters:
    -----------
    input_path : str
        The path to the input CSV file containing the data.
    output_path : str
        The path to the output CSV file where the transformed data will be saved.
    mode : str
        The mode of data transformation. Valid values are 'train' and any other value for
        other transformations.

    Methods:
    ---------
    read_data()
        Reads a CSV file from the path specified in self.input_path and returns a pandas
        DataFrame.

    data_transformation(df, mode)
        Perform data transformation on the input DataFrame based on the specified mode.

    write_prepared_data(transformed_dataframe)
        Write the transformed DataFrame to a CSV file.

    run()
        Execute the entire data preparation pipeline.
    """

    def __init__(self, input_path, output_path, mode):
        self.mode = mode
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
        logging.info("Pandas DataFrame read.")
        return pandas_df

    def data_transformation(self, df, mode):
        """
        Perform data transformation on the input DataFrame based on the specified mode.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing the data for transformation.
        mode : str
            The mode of transformation. Should be either 'train' or another value for
            other transformations.

        Returns:
        --------
        pandas.DataFrame
            Transformed DataFrame after applying data preprocessing steps.

        Description:
        ------------
        This method conducts data transformation by splitting the input DataFrame into
        training and testing sets, selecting specific features, performing preprocessing,
        and creating pipelines to handle numeric and categorical data separately.

        The 'train' mode fits the transformation pipeline on the training data and applies
        it to either the training or testing set based on the mode. The transformed DataFrame
        is returned with modified feature names.

        For modes other than 'train', the transformation pipeline is fitted on the training
        data and applied to the test set, returning the transformed DataFrame.

        Steps:
        ------
        1. Splits the input DataFrame into train and test sets.
        2. Selects numeric, categorical, and metadata features.
        3. Cleans the data by dropping NA values.
        4. Creates separate pipelines for numeric and categorical features.
        5. Uses ColumnTransformer to apply pipelines to specific feature groups.
        6. Integrates all steps into a comprehensive pipeline.
        7. Transforms the data based on the specified mode.

        Note:
        -----
        'numeric__', 'categoric__', and 'remainder__' prefixes are removed from the
        transformed column names for better readability.

        """
        # Split the data frame
        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=17)
        test_df.to_csv('./data/test.csv', sep=',', index=False)
        logging.info("Test DataFrame created after split.")

        # Choosing the features to use
        numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars',
                            'TotalBsmtSF', 'FullBath', 'YearRemodAdd',
                            'YearBuilt', 'Fireplaces', 'MasVnrArea', 'LotArea',
                            'SalePrice']
        categorical_features = ['ExterQual', 'CentralAir']
        metadata_features = ['Id']  # noqa E221

        # Final features
        final_features = numeric_features + categorical_features + metadata_features   # noqa E501

        # Removing columns
        data_cleaned = train_df[final_features]
        data_cleaned_test = test_df[final_features]

        # Removing na
        data_cleaned.dropna(inplace=True)
        data_cleaned_test.dropna(inplace=True)

        # Combine all the preprocessing steps into a single pipeline for
        # numeric features
        numeric_pipeline = Pipeline([
            # Step 1: Impute missing values
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Step 2: Scale the data
            # Step 3: Apply the CappingTransformer
            ('capper', CappingTransformer(threshold=1.5))
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            # Step 1: Impute missing values
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # Step 2: One-hot encode categorical features
            ('onehot', OneHotEncoder())
        ])

        # Create a ColumnTransformer to apply the numeric_pipeline to numeric features and the
        # categorical_pipeline to categorical features   # noqa E501
        feature_transformer = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric_features),
                ('categoric', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough'  # Pass through features not specified in transformers
        )

        # Combine all the steps into a single pipeline
        full_pipeline = Pipeline([
            ('feature_transform', feature_transformer)
        ])

        # Flag the mode
        if mode == 'train':
            full_pipeline.fit(X=data_cleaned[final_features])
            final_data = full_pipeline.transform(
                X=data_cleaned[final_features])
            final_data_df = pd.DataFrame(final_data,
                                         columns=full_pipeline.get_feature_names_out())
            df_transformed = final_data_df.rename(
                columns=lambda x: x.replace(
                    'numeric__',
                    '') .replace(
                    'categoric__',
                    '') .replace(
                    'remainder__',
                    ''))
            logging.info("Train transformation pipeline complete.")
        else:
            full_pipeline.fit(X=data_cleaned[final_features])
            final_data = full_pipeline.transform(
                X=data_cleaned_test[final_features])
            final_data_df = pd.DataFrame(final_data,
                                         columns=full_pipeline.get_feature_names_out())
            df_transformed = final_data_df.rename(
                columns=lambda x: x.replace(
                    'numeric__',
                    '') .replace(
                    'categoric__',
                    '') .replace(
                    'remainder__',
                    ''))
            logging.info("Test transformation complete.")

        return df_transformed

    def write_prepared_data(self, transformed_dataframe):
        """
        Write the transformed DataFrame to a CSV file.

        Parameters:
        -----------
        transformed_dataframe : pandas.DataFrame
            Transformed DataFrame that needs to be written to a CSV file.

        Description:
        ------------
        This method writes the provided transformed DataFrame to a CSV file specified
        by the output path set within the class instance.

        The transformed_dataframe is saved as a CSV file without including the index.

        """
        transformed_dataframe.to_csv(self.output_path, index=False)
        logging.info("Transformed file complete.")

    def run(self):
        """
        Execute the entire data preparation pipeline.

        Description:
        ------------
        This method orchestrates the entire data preparation pipeline by:
        1. Reading data from the specified input path.
        2. Performing data transformation on the read data based on a specified mode.
        3. Writing the transformed data to a CSV file at the specified output path.

        The 'run' method integrates the 'read_data', 'data_transformation', and
        'write_prepared_data' methods to facilitate a complete data preparation process.

        """
        df = self.read_data()
        df_transformed = self.data_transformation(df, mode)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Execution mode: train or test')
    args = parser.parse_args()

    # Retrieve the specified mode from command-line arguments
    mode = args.mode

    # Set the base directory to the parent directory of the current file
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    # Start Log
    logging.info('feature_engineering.py starting.')

    # Define input and output paths based on the specified mode
    if mode == 'train':
        # Define paths for training mode
        IN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
        OUT_PATH = os.path.join(BASE_DIR, 'data', 'transformed_dataframe.csv')
    else:
        # Define paths for test mode
        IN_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
        OUT_PATH = os.path.join(BASE_DIR, 'data', 'test_df_transformed.csv')

    # Initialize and execute the Feature Engineering Pipeline based on the mode
    FeatureEngineeringPipeline(
        input_path=IN_PATH,
        output_path=OUT_PATH,
        mode=mode).run()

    # End Log
    logging.info('feature_engineering.py end.')
