import re
import datetime
import pandas as pd
from osisoft.pidevclub.piwebapi.pi_web_api_client import PIWebApiClient
from osisoft.pidevclub.piwebapi.rest import ApiException
import time

def armar_encabezado(client, paths):
    webs_id = []
    for path in paths:
        try:
            point = client.point.get_by_path(path[3:])
            webs_id.append(point.web_id)
        except Exception as e:
            print(f"Error retrieving information for path {path}: {e}")

    return point, webs_id

def extraer_valores(client, webs_id):
    current_values = []
    for web_id in webs_id:
        current_values.append(client.stream.get_value(web_id=web_id).value)

    time = client.stream.get_value(web_id=webs_id[0]).timestamp  # Usar el primer web_id

    return current_values, time

def formato_df(paths, current_values, time):
    df_current_values = pd.DataFrame([current_values], columns=paths)
    df_time = pd.DataFrame([time], columns=['Timestamp'])
    df_current_values.columns = df_current_values.columns.str.replace(
        'pi:\\\CAMHIS01\\', '')
    df_final = pd.concat([df_time, df_current_values],
                         axis=1)
    df_final.set_index('Timestamp', inplace=True)

    return df_final

def localize(df_final: pd.DataFrame):
    df_final_hora = df_final.copy()
    df_final_hora.index = pd.to_datetime(df_final_hora.index).tz_convert(
        'America/Argentina/Buenos_Aires').tz_localize(None)

    return df_final_hora

def main():
    client = PIWebApiClient("https://piwebapi.cammesa.com/piwebapi",
                            useKerberos=False, verifySsl=True)

    df_paths = pd.read_csv('paths_reducido.csv')
    paths = df_paths['Paths'].tolist()

    while True:
        # Llamada a la función armar_encabezado solo la primera vez
        if 'point' not in locals():  # Verificar si 'point' no está en las variables locales
            point, webs_id = armar_encabezado(client, paths)

        current_values, time = extraer_valores(client, webs_id)
        df = formato_df(paths, current_values, time)
        df_nea_final = localize(df)
        print("Nuevo valor")
        df_nea_final.to_pickle('df_nea_final.pkl')
        df_nea_final.to_csv('df_nea_final.csv')

if __name__ == '__main__':
    import threading

    timer = threading.Timer(1*60, main)
    timer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        timer.cancel()
     
    # main()
