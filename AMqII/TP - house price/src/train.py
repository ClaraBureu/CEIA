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
import matplotlib.pyplot as plt
import seaborn as sns


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
        train_df = df
        # Load and split the data
        target_variable = 'SalePrice'

        X_train, X_val, y_train, y_val = load_and_split_data(
            train_df, target_variable)

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
        
        X_train_final = train_df.drop(columns=target_variable)
        y_train_final = train_df[target_variable]
        
        best_model = best_models["GradientBoostingRegressor"][0]
        best_model.fit(X=X_train_final, y=y_train_final)
        y_train_final_pred = best_model.predict(X=X_train_final)

        # Calculate RMSE for the training set
        rmse_train = mean_squared_error(
            y_train_final, y_train_final_pred, squared=False)

        # Create a bar plot
        model_names = ['Training RMSE']
        rmse_scores = [rmse_train]

        # Create a bar plot using Seaborn
        plt.figure(figsize=(8, 5))
        sns.set(style="whitegrid")

        # Create the bar plot
        ax = sns.barplot(x=model_names, y=rmse_scores, palette="Blues")

        plt.xlabel('RMSE Type')
        plt.ylabel('RMSE Score')
        plt.title('RMSE Comparison for Training and Test Sets')
        plt.xticks(rotation=0)  # Keep x-axis labels horizontal

        # Add RMSE values as labels in each column
        for i, v in enumerate(rmse_scores):
            ax.text(i, v, f'RMSE: {v:.2f}', ha='center', va='bottom', fontsize=10)

        # Show the plot
        plt.show()
        
        return best_model

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
                          model_path='..\\data\\model.pkl').run()