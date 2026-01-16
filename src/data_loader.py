import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file):
    # wczytywanie csv excel json
    # cache streamlita zeby bylo szybciej
    try:
        filename = file.name
        if filename.endswith('.csv'):
            return pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(file)
        elif filename.endswith('.json'):
            return pd.read_json(file)
        else:
            st.error("Nieobsługiwany format pliku")
            return None
    except Exception as e:
        st.error(f"Błąd podczas parsowania pliku: {e}")
        return None