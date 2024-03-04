import re
import datetime
import pandas as pd
from osisoft.pidevclub.piwebapi.pi_web_api_client import PIWebApiClient
from osisoft.pidevclub.piwebapi.rest import ApiException
import pickle
import streamlit as st
import threading
import time

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

def predict_real_time(data):

    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    return prediction, probability


# Streamlit app
def main():
    st.title("Aplicación para detectar colapsos")


def update_predictions():
    print("Actualizando valores")
    input_data = pickle.load(open('df_nea_final.pkl', 'rb'))
    # Make predictions
    prediction, probability = predict_real_time(input_data)

    if prediction == 1:
        text = "Hay peligro de colapso en el NEA"
    else:
        text = "No hay peligro de colapso en el NEA"


    # Display results
    st.subheader(text)
    st.write(f"Predicted Class: {prediction[0]}")
    st.write(f"Probability: {probability[0]:.2f}")


if __name__ == '__main__':

    main()
    try:
        while True:
            update_predictions()
            time.sleep(60)
    except KeyboardInterrupt:
        print("Program terminated by user.")
