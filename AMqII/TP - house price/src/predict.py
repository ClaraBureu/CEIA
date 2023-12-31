"""
predict.py

This script defines a MakePredictionPipeline class to facilitate the process of making predictions
using a pre-trained model on input data and writing the results to a CSV file.

Imports:
- os
- pickle as pkl
- pandas as pd
- DataFrame from pandas

Classes:
- MakePredictionPipeline: A class for loading data, loading a pre-trained model,
  making predictions, and writing the results to a CSV file.

Usage:
- Execute this script to make predictions using a pre-trained model on specified input data.
- Ensure the necessary input files ('test_df_transformed.csv' and 'model.pkl') exist 
in the 'data' directory.

DESCRIPCIÓN: predict.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

# Imports
import logging
import os
import pickle as pkl
import pandas as pd
from pandas import DataFrame

# Log setting
logging.basicConfig(level=logging.DEBUG,
                    filename='data_logger.log',
                    filemode='a',
                    format='%(asctime)s:%(levelname)s:%(message)s')


class MakePredictionPipeline(object):
    """
    A class for making predictions using a trained model.

    This class provides methods for loading data, loading a trained model,
    making predictions, and writing predictions to a CSV file.

    Parameters:
    -----------
    input_path : str
        The path to the input CSV file containing the data for prediction.
    output_path : str
        The path to the output CSV file where the predictions will be saved.
    model_path : str, optional
        The path to the trained model file. If None, the model is loaded from 
        the class attribute 'self.model'.

    Methods:
    ---------
    load_data()
        Reads a CSV file from the path specified in self.input_path and
        returns a pandas DataFrame.

    load_model()
        Load the inference model from the specified path.

    make_predictions(data, model=None)
        Generate predictions using the loaded model.

    write_predictions(predicted_dataframe)
        Write the predicted results to a CSV file.

    run()
        Execute the complete prediction pipeline.
    """

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

    def load_data(self):
        """
        Reads a CSV file from the path specified in self.input_path and
        returns a pandas DataFrame.

        Returns:
        pandas.DataFrame: A pandas DataFrame object containing the data from
        the CSV file.
        """
        pandas_df = pd.read_csv(self.input_path)
        logging.info('Pandas DataFrame loaded.')
        return pandas_df

    def load_model(self) -> None:
        """
        Load the inference model from the specified path.

        Returns:
        --------
        None

        Description:
        ------------
        This method loads the trained inference model from the specified file path
        using pickle (pkl) deserialization.

        The loaded model is stored within the class attribute 'self.model' for further
        usage in inference tasks.

        Note:
        -----
        Ensure the 'model_path' attribute is correctly set before calling this method
        to load the model.

        """
        with open(self.model_path, 'rb') as file:
            self.model = pkl.load(file)
        logging.info('Model loaded.')

    def make_predictions(self, data, model=None):
        """
        Generate predictions using the loaded model.

        Parameters:
        -----------
        data : array-like or DataFrame
            Input data for making predictions.
        model : object, optional
            Custom model object. If None, the internal model stored within the class is used.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the predictions generated by the model.

        Raises:
        -------
        Exception
            Raised when the model is not loaded.

        Description:
        ------------
        This method generates predictions using the loaded model on the provided input data.

        If a custom model object is not provided, it uses the model stored within the class.
        If the model is not loaded, an Exception is raised to indicate that the model
        needs to be loaded before making predictions.

        The input 'data' can be an array-like object or a pandas DataFrame compatible
        with the prediction method of the stored model.

        """
        if self.model is None:
            raise Exception("Model not loaded")

        new_data = pd.DataFrame(self.model.predict(data))
        logging.info('Predictions generated.')

        return new_data

    def write_predictions(self, predicted_dataframe: DataFrame) -> None:
        """
        Write the predicted results to a CSV file.

        Parameters:
        -----------
        predicted_dataframe : pandas.DataFrame
            DataFrame containing the predicted results.

        Returns:
        --------
        None

        Description:
        ------------
        This method writes the predicted results stored in the provided DataFrame to a CSV file
        specified by the output path within the class instance.

        The predicted results are saved as a CSV file without including the index.

        """
        predicted_dataframe.to_csv(self.output_path, index=False)
        logging.info('Predicted File generated.')

    def run(self):
        """
        Execute the complete prediction pipeline.

        Description:
        ------------
        This method orchestrates the entire prediction pipeline by:
        1. Loading the necessary data for prediction.
        2. Preparing the data by removing the 'SalePrice' column.
        3. Loading the model for prediction.
        4. Generating predictions using the loaded model on the prepared data.
        5. Writing the predicted results to a CSV file.

        The 'run' method integrates the 'load_data', 'load_model', 'make_predictions',
        and 'write_predictions' methods to facilitate a complete prediction process.

        """
        data = self.load_data()
        data_for_prediction = data.drop(columns=['SalePrice'])
        model = self.load_model()
        predictions = self.make_predictions(data_for_prediction, model)

        self.write_predictions(predictions)


if __name__ == "__main__":
    # Get the base directory of the current script file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Start Log
    logging.info('predict.py starting.')

    # Initialize the prediction pipeline with input, output, and model paths
    pipeline = MakePredictionPipeline(
        input_path=os.path.join(BASE_DIR, '..', 'data', 'test_df_transformed.csv'),
        output_path=os.path.join(BASE_DIR, '..', 'data', 'predicted_dataframe.csv'),
        model_path=os.path.join(BASE_DIR, '..', 'data', 'model.pkl'))

    # Execute the prediction pipeline
    pipeline.run()

    # End Log
    logging.info('predict.py end.')
