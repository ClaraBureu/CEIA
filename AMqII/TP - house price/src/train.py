"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: train.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 18/11/2023
"""

# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from utils import load_and_split_data
from utils import train_regressor
import pickle


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
        return pandas_df

    def model_training(self, df: pd.DataFrame):
        """
        COMPLETAR DOCSTRING
        
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
            model_trained = train_regressor(model, X_train, y_train, params)

            # Calculate RMSE for the training set
            y_train_pred = model_trained.predict(X_train)
            rmse_train = mean_squared_error(
                y_train, y_train_pred, squared=False)

            # Calculate RMSE for the validation set
            y_val_pred = model_trained.predict(X_val)
            rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

            # Store the best model and RMSE scores in the dictionary
            best_models[model.__class__.__name__] = (
                model_trained, y_val_pred, rmse_val, y_train_pred, rmse_train)
        
        return model_trained

    def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        try:
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_trained, file)
        except Exception as e:
            print('Error dumping the model: {}'.format(e))
  
        return None
    
    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


if __name__ == "__main__":

    ModelTrainingPipeline(input_path='..\\data\\transformed_dataframe.csv',
                          model_path='..\\data\\model.plk').run()