"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: predict.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 18/11/2023
"""
# Imports
import pickle as pkl
import pandas as pd
from pandas import DataFrame


class MakePredictionPipeline(object):
   
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
        return pandas_df

    def load_model(self) -> None:
        """
        Cargar el modelo de inferencia.
        """
        with open(self.model_path, 'rb') as file:
            self.model = pkl.load(file)

        return None

    def make_predictions(self, data):
        """
        Realizar las predicciones sobre el dataset de entrada.
        """

        if self.model is None:
            raise Exception("Model not loaded")

        new_data = pd.DataFrame(self.model.predict(data))

        return new_data

    def write_predictions(self, predicted_dataframe: DataFrame) -> None:
        """
        Guardar las predicciones en el path de salida.
        """
        predicted_dataframe.to_csv(self.output_path, index=False)
        return None

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
   
    pipeline = MakePredictionPipeline(
        input_path=r'..\\data\transformed_dataframe.csv',
        output_path=r'..\\data\predicted_dataframe.csv',
        model_path=r'..\\data\model.pkl')
    pipeline.run()