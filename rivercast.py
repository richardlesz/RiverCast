import streamlit as st
import pandas as pd
import numpy as np

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
    st.session_state.all_stations = pd.concat([df_Mitchell_Snowy, df_Thomson, df_Mitta_King], ignore_index=True)
    st.session_state.all_stations = st.session_state.all_stations.drop(columns=['Crossing (m)', 'Recent Data'])
    st.session_state.stations = pd.concat([df_Mitchell_Snowy.loc[[18,46]], df_Thomson.loc[[15]], df_Mitta_King.loc[[24,28,57]]], ignore_index=True)

    releases = pd.read_csv('./data/Jindabyne-Dam-release-targets-all.csv')
    releases['date'] = pd.to_datetime(releases['date'])

    st.session_state.predictions = releases.copy()
    st.session_state.predictions['target_T-3'] = st.session_state.predictions['target'].shift(3)

    today = pd.Timestamp.today().normalize()
    st.session_state.predictions = st.session_state.predictions[st.session_state.predictions['date'] >= today]
    st.session_state.predictions['waterlevel_prediction'] = pd.NA
    st.session_state.predictions.iloc[0, 3] = st.session_state.stations.iloc[0, 3]  # Set the first water level prediction to the current water level
    st.session_state.predictions.iloc[0, 3] = pd.to_numeric(st.session_state.predictions.iloc[0, 3], errors='coerce')  # Convert to numeric, coercing errors to NaN

    for i in range(1, len(st.session_state.predictions)):
        features = [
            0,  # Placeholder for rainfall_value_T-2 (not used in prediction)
            st.session_state.predictions.iloc[i]['target_T-3'],  # release_target_T-3
            st.session_state.predictions.iloc[i-1]['waterlevel_prediction']  # waterlevel_value_T-1
        ]
        st.session_state.predictions.iloc[i, 3] = waterlevel_predictor.predict(features)

st.set_page_config(page_title="RiverCast", page_icon=":droplet:", layout="wide")

st.title("RiverCast")

col1, col2, col3 = st.columns([1,3,1])

with col1:
    for station in st.session_state.stations.itertuples():
        st.metric(station._1, station._4, station.Tendency, delta_arrow="off")

with col2:
    option = st.selectbox("Select a station to view predictions", st.session_state.stations['Station Name'])
    st.header(f"{option}")

    if option == "Snowy River at McKillops Bridge":
        st.subheader("ML Predicted Water Level", text_alignment='center')
        st.line_chart(st.session_state.predictions, x='date', y='waterlevel_prediction', x_label='Date', y_label='Predicted Water Level (m)')
        st.subheader("Jindabyne Dam Release Targets", text_alignment='center')
        st.line_chart(st.session_state.predictions, x='date', y='target', x_label='Date', y_label='Release Target (ML/day)')
    else:
        st.subheader("Coming Soon")
        st.write("This station is not yet supported. We are working on adding more stations and predictions in the near future. Stay tuned!")
        st.write("This graph is a placeholder and does not represent actual predictions for this station. We will update it with real data as soon as it's available.")
        st.subheader("ML Predicted Water Level", text_alignment='center')
        dates = pd.date_range(start="2026-06-14", periods=365, freq="D")
        df = pd.DataFrame({"Date": dates, "Random_Value": np.random.rand(365)})
        st.line_chart(df, x='Date', y='Random_Value', x_label='Date', y_label='Predicted Water Level (m)')


st.session_state.all_stations