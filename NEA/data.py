import re
import datetime
import pandas as pd
from osisoft.pidevclub.piwebapi.pi_web_api_client import PIWebApiClient
from osisoft.pidevclub.piwebapi.rest import ApiException

def data_tiemporeal():
    df_paths = pd.read_csv('paths.csv')
    paths = df_paths['Paths'].tolist()

    client = PIWebApiClient("https://piwebapi.cammesa.com/piwebapi",
                                useKerberos=False, verifySsl=True)

    df_nea = pd.DataFrame()
    current_datetime = datetime.datetime.now()
    start_datetime = current_datetime - datetime.timedelta(minutes=1)  # Adjust the time window as needed

    try:
        df_interpolate = client.data.get_multiple_recorded_values(
            paths=paths,
            start_time=start_datetime,
            end_time=current_datetime,
            # interval="1m"
        )

        if df_interpolate.empty or not all(col in df_interpolate.columns for col in ['Timestamp', 'Value']):
            print("Tags not found or data is missing. Skipping...")

        df_nea = df_interpolate.set_index('Timestamp')

    except ApiException as e:
        if e.status == 404:
            print("One or more tags not found. Skipping...")
        else:
            # Handle other ApiException or re-raise the exception if needed
            raise
    
    df_nea.to_csv('df_nea.csv')
    return df_nea

def main(fecha=None):
    df_data_tr = data_tiemporeal()

if __name__ == '__main__':
    main()
