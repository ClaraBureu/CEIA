import re
import datetime
import pandas as pd
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
    st.set_page_config(
        page_title="Aplicación para detectar colapsos",
        page_icon="C:/Users/CBureu/Documents/IOP/NEA/Graficas/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Aplicación para detectar colapsos")

    # Checkbox for real-time data
    real_time_option = st.checkbox("Consultar en tiempo real")

    index = st.session_state.get('index', 0)

    if real_time_option:
        st.write("Datos en tiempo real")
    else:
        st.write("Datos de test")

    return real_time_option, index

def update_predictions(real_time_option, index):
    print("Actualizando valores")

    if real_time_option:
        # Fetch real-time data
        # Modify this part based on how you retrieve your real-time data
        input_data = real_time_data()
    else:
        # Fetch historical data based on the selected date range
        # Modify this part based on how you retrieve your historical data
        df_test = test_data()
        input_data = df_test.iloc[[index]]
        index += 1
        print(input_data, index, input_data.index)

    # Make predictions
    prediction, probability = predict_real_time(input_data)

    if prediction == 1:
        text = "Hay peligro de colapso en el NEA"
    else:
        text = "No hay peligro de colapso en el NEA"

    # Display results
    st.subheader(text)
    st.write(f"Fecha: {input_data.index[0]}")
    st.write(f"Predicted Class: {prediction[0]}")
    st.write(f"Probability: {probability[0]:.2f}")

    st.session_state.index = index

    return index

def real_time_data():
    # En este caso, simplemente devolvemos el DataFrame de prueba
    # pero podrías modificar esta función para conectarte a una fuente en tiempo real
    input_data = pickle.load(open('df_nea_final.pkl', 'rb'))
    return input_data

def test_data():
    # Cargamos el DataFrame de prueba
    df_test = pickle.load(open('df_test.pkl', 'rb'))

    return df_test

if __name__ == '__main__':
    time_option, index = main()

    try:
        while True:
            index = update_predictions(time_option, index)
            time.sleep(60)
    except KeyboardInterrupt:
        print("Program terminated by user.")
