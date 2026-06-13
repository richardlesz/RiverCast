import streamlit as st
import pandas as pd

import waterlevel_predictor

def fetch_bom_data():
    url_Mitchell_Snowy = "ftp://ftp.bom.gov.au/anon/gen/fwo/IDV60078.html"
    url_Thomson = "ftp://ftp.bom.gov.au/anon/gen/fwo/IDV60079.html"
    url_Mitta_King = "ftp://ftp.bom.gov.au/anon/gen/fwo/IDV60151.html"

    tables_Mitchell_Snowy = pd.read_html(url_Mitchell_Snowy)
    tables_Thomson = pd.read_html(url_Thomson)
    tables_Mitta_King = pd.read_html(url_Mitta_King)

    df_Mitchell_Snowy = tables_Mitchell_Snowy[0]
    df_Thomson = tables_Thomson[0]
    df_Mitta_King = tables_Mitta_King[0]

    st.session_state.fetched_data = True

    return df_Mitchell_Snowy, df_Thomson, df_Mitta_King

if 'fetched_data' not in st.session_state:
    df_Mitchell_Snowy, df_Thomson, df_Mitta_King = fetch_bom_data()
    all_stations = pd.concat([df_Mitchell_Snowy, df_Thomson, df_Mitta_King], ignore_index=True)
    stations = pd.concat([df_Mitchell_Snowy.loc[[18,46]], df_Thomson.loc[[15]], df_Mitta_King.loc[[24,28,57]]], ignore_index=True)

    releases = pd.read_csv('./data/Jindabyne-Dam-release-targets-all.csv')
    releases['date'] = pd.to_datetime(releases['date'])

    predictions = releases.copy()
    predictions['target_T-3'] = predictions['target'].shift(3)

    today = pd.Timestamp.today().normalize()
    predictions = predictions[predictions['date'] >= today]
    predictions['waterlevel_prediction'] = pd.NA
    predictions.iloc[0, 3] = stations.iloc[0, 3]  # Set the first water level prediction to the current water level
    predictions.iloc[0, 3] = pd.to_numeric(predictions.iloc[0, 3], errors='coerce')  # Convert to numeric, coercing errors to NaN

    for i in range(1, len(predictions)):
        features = [
            0,  # Placeholder for rainfall_value_T-2 (not used in prediction)
            predictions.iloc[i]['target_T-3'],  # release_target_T-3
            predictions.iloc[i-1]['waterlevel_prediction']  # waterlevel_value_T-1
        ]
        predictions.iloc[i, 3] = waterlevel_predictor.predict(features)

st.set_page_config(page_title="RiverCast", page_icon=":droplet:", layout="wide")

st.title("RiverCast")

col1, col2, col3 = st.columns([1,3,1])

with col1:
    for station in stations.itertuples():
        st.metric(station._1, station._4, station.Tendency)

with col2:
    st.line_chart(predictions, x='date', y='waterlevel_prediction', x_label='Date', y_label='Predicted Water Level (m)')
    st.line_chart(predictions, x='date', y='target', x_label='Date', y_label='Release Target (ML/day)')

all_stations