"""
train.py

This script defines a ModelTrainingPipeline class responsible for training regression models
and serializing the best-performing model for future use.

Imports:
- os
- pandas as pd
- LinearRegression, DecisionTreeRegressor, GradientBoostingRegressor from sklearn
- mean_squared_error from sklearn.metrics
- load_and_split_data, train_regressor from utils
- pickle for serialization

Classes:
- ModelTrainingPipeline: A class for reading data, training regression models, and serializing the best model.

Usage:
- Execute this script to read transformed data and train a regression model, saving it as 'model.pkl'.
- Ensure the transformed data file ('transformed_dataframe.csv') exists in the 'data' directory.

DESCRIPCIÓN: train.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

# Imports
import logging
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from utils import load_and_split_data
from utils import train_regressor
import pickle

# Log setting
logging.basicConfig(level=logging.DEBUG, 
                    filename='data_logger.log', 
                    filemode='a', 
                    format='%(asctime)s:%(levelname)s:%(message)s')

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads a CSV file from the path specified in self.input_path and
        returns a pandas DataFrame.

        Returns:
        pandas.DataFrame: A pandas DataFrame object containing the data from
        the CSV file.
        """
        pandas_df = pd.read_csv(self.input_path)
        logging.info('Pandas DataFrame read.')
        return pandas_df

    def model_training(self, df: pd.DataFrame):
        """
        Train multiple regression models and select the best one based on RMSE.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the training data.

        Returns:
        --------
        object
            The trained regression model with the best performance.

        Description:
        ------------
        This method performs training on various regression models using the provided DataFrame.
        It evaluates each model's performance based on Root Mean Squared Error (RMSE) on both
        the training and validation sets and selects the best-performing model.

        The models trained include Linear Regression, Decision Tree Regression, and Gradient
        Boosting Regression with predefined hyperparameter grids.

        The best-performing model based on the validation set RMSE is stored and finally, retrained
        on the entire dataset (without validation splitting) for potential future use.

        """
        # Load and split the data
        target_variable = 'SalePrice'

        X_train, X_val, y_train, y_val = load_and_split_data(
            df, target_variable)

        # Define hyperparameter grids for each model
        model_params = {
            LinearRegression(): {},
            DecisionTreeRegressor(random_state=42): {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            GradientBoostingRegressor(random_state=42): {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }

        # Store the best models and RMSE scores in a dictionary
        best_models = {}

        for model, params in model_params.items():
            best_model = train_regressor(model, X_train, y_train, params)

            # Calculate RMSE for the training set
            y_train_pred = best_model.predict(X_train)
            rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

            # Calculate RMSE for the validation set
            y_val_pred = best_model.predict(X_val)
            rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

            # Store the best model and RMSE scores in the dictionary
            best_models[model.__class__.__name__] = (
                best_model, y_val_pred, rmse_val, y_train_pred, rmse_train)
        
        X_train_final = df.drop(columns=target_variable)
        y_train_final = df[target_variable]
        
        best_model = best_models["GradientBoostingRegressor"][0]
        best_model.fit(X=X_train_final, y=y_train_final)
        y_train_final_pred = best_model.predict(X=X_train_final)
        
        logging.info('Best model found.')
        return best_model

    def model_dump(self, model_trained) -> None:
        """
        Serialize and save the trained model to a file.

        Parameters:
        -----------
        model_trained : object
            The trained model object to be serialized and saved.

        Returns:
        --------
        None

        Description:
        ------------
        This method serializes the trained model object using pickle and saves it
        to a file specified by the 'model_path' attribute within the class instance.

        If successful, the model is saved as a serialized object in the specified file.
        In case of any error during the serialization process, an exception is caught,
        and an error message is printed.

        """
        try:
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_trained, file)
            logging.info('Model.pkl created.')
        except Exception as e:
            print('Error dumping the model: {}'.format(e))
            logging.warning(f'Error dumping the model: {e}')
            
    
    def run(self):
        """
        Execute the complete training and model serialization process.

        Description:
        ------------
        This method orchestrates the entire process of:
        1. Reading data from a source.
        2. Training a regression model using the read data.
        3. Serializing and saving the trained model to a file.

        The 'run' method integrates the 'read_data', 'model_training', and 'model_dump'
        methods to execute the complete training and serialization process.

        """
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


if __name__ == "__main__":
    # Get the base directory of the current script file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Starting log
    logging.info('train.py starting.')

    # Initialize and execute the ModelTrainingPipeline
    # Load transformed data from a file and train a model
    ModelTrainingPipeline(input_path=os.path.join(BASE_DIR,'..', 'data','transformed_dataframe.csv'),
                          model_path=os.path.join(BASE_DIR,'..', 'data', 'model.pkl')).run()
    
    # Log end
    logging.info('train.py end.')