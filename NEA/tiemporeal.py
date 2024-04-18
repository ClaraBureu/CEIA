import re
import datetime
import pandas as pd
import pickle
import streamlit as st
import threading
import time

model = pickle.load(open('model.pkl', 'rb'))

def predict_real_time(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:,0]

    return prediction, probability

def main():

    st.title("Aplicación para detectar colapsos")

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

        input_data = real_time_data()
    else:

        df_test = test_data()
        input_data = df_test.iloc[[index]]
        index += 1
        print(input_data, index, input_data.index)

    prediction, probability = predict_real_time(input_data)

    if prediction == 1:
        text = "Hay peligro de colapso en el NEA"
    else:
        text = "No hay peligro de colapso en el NEA"

    st.subheader(text)
    # st.write("Información:")
    # st.write(input_data) 

    results_df = pd.DataFrame({
        "Fecha": [input_data.index[0]],
        "Predicción de la clase": [prediction[0]],
        "Probabilidad de colapso": [f"{probability[0]:.2f}"]
    })

    st.write("Información:")
    st.write(results_df)

    st.session_state.index = index

    return index

def real_time_data():

    input_data = pickle.load(open('df_nea_final.pkl', 'rb'))
    return input_data

def test_data():

    #df_test = pickle.load(open('df_test.pkl', 'rb'))
    df_test = pickle.load(open('colapso_15032024.pkl', 'rb'))

    return df_test

if __name__ == '__main__':
    time_option, index = main()

    try:
        while True:
            index = update_predictions(time_option, index)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
