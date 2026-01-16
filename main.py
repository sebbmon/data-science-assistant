import streamlit as st
import pandas as pd
import os

from src.data_loader import load_data
from src.visualization import plot_histogram, plot_heatmap, plot_scatter
from src.preprocessing import Preprocessor
from src.model_builder import ModelBuilder
from src.evaluation import ModelEvaluator

st.set_page_config(page_title="Data Science Assistant", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

st.sidebar.title("Nawigacja")
step = st.sidebar.radio("Wybierz etap:", 
    ["1. Wczytaj Dane", "2. Eksploracja (EDA)", "3. Preprocessing", "4. Budowa Modelu", "5. Ocena i Eksport"])

if step == "1. Wczytaj Dane":
    st.header("Wczytaj Zbiór Danych")
    uploaded_file = st.file_uploader("Wybierz plik (CSV, Excel, JSON)", type=['csv', 'xlsx', 'json'])
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            
            st.success("Plik wczytany pomyślnie!")
            st.write("Podgląd surowych danych:", df.head())
            st.write(f"Wymiary: {df.shape}")
        except Exception as e:
            st.error(f"Błąd wczytywania: {e}")

elif step == "2. Eksploracja (EDA)":
    st.header("Eksploracja Danych")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("Statystyki opisowe")
        st.write(df.describe())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Typy danych:", df.dtypes)
        with col2:
            st.write("Braki danych (NaN):", df.isnull().sum())

        st.divider()
        
        st.subheader("Wizualizacje")
        viz_type = st.selectbox("Wybierz wykres", ["Histogram", "Scatter Plot", "Mapa Korelacji"])
        
        if viz_type == "Histogram":
            col = st.selectbox("Wybierz kolumnę", df.columns)
            st.pyplot(plot_histogram(df, col))
            
        elif viz_type == "Scatter Plot":
            col_x = st.selectbox("Oś X", df.columns)
            col_y = st.selectbox("Oś Y", df.columns, index=1 if len(df.columns) > 1 else 0)
            st.pyplot(plot_scatter(df, col_x, col_y))
            
        elif viz_type == "Mapa Korelacji":
            st.pyplot(plot_heatmap(df))
            
    else:
        st.warning("Najpierw wgraj dane w etapie 1.")

elif step == "3. Preprocessing":
    st.header("Przygotowanie Danych")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        target = st.selectbox("Wybierz zmienną docelową (y)", df.columns)
        st.session_state.target_col = target
        
        st.subheader("Konfiguracja czyszczenia")
        
        missing_strategy = st.selectbox("Obsługa braków danych", ["Brak", "Usuń wiersze", "Średnia", "Mediana", "Moda"])
        encoding_method = st.selectbox("Kodowanie zmiennych kategorycznych", ["Label Encoding", "One-Hot Encoding"])
        scaling_method = st.selectbox("Skalowanie cech numerycznych", ["Brak", "StandardScaler", "MinMaxScaler"])
        test_size = st.slider("Wielkość zbioru testowego (%)", 10, 50, 20) / 100.0

        if st.button("Przetwórz dane"):
            processor = Preprocessor(target_col=target)
            
            df_clean = processor.handle_missing_values(df, strategy=missing_strategy)
            
            df_encoded = processor.encode_categorical(df_clean, method=encoding_method)
            
            if scaling_method != "Brak":
                df_encoded = processor.scale_features(df_encoded, method=scaling_method)
            
            st.session_state.df_processed = df_encoded
            
            X_train, X_test, y_train, y_test = processor.split_data(df_encoded, test_size=test_size)
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Dane przetworzone pomyślnie")
            st.write("Podgląd po przetworzeniu:", df_encoded.head())
            st.write(f"Kształt treningowy: {X_train.shape}, Kształt testowy: {X_test.shape}")
            
    else:
        st.warning("Najpierw wgraj dane w etapie 1.")

elif step == "4. Budowa Modelu":
    st.header("Trenowanie Modelu")
    
    if st.session_state.get('X_train') is not None:
        
        task_type = st.radio("Rodzaj problemu", ["Klasyfikacja", "Regresja"])
        
        builder = ModelBuilder()
        model_names = builder.get_available_models(task_type)
        selected_model = st.selectbox("Wybierz algorytm", model_names)
        
        use_grid_search = st.checkbox("Użyj Grid Search (Optymalizacja hiperparametrów)")
        
        if st.button("Trenuj Model"):
            with st.spinner("Trenowanie modelu... to może chwilę potrwać."):
                try:
                    model, params = builder.train_model(
                        st.session_state.X_train, 
                        st.session_state.y_train, 
                        task_type, 
                        selected_model,
                        use_grid_search
                    )
                    
                    st.session_state.model = model
                    st.session_state.task_type = task_type
                    
                    st.success(f"Model {selected_model} wytrenowany!")
                    if use_grid_search:
                        st.write("Najlepsze parametry:", params)
                except Exception as e:
                    st.error(f"Błąd podczas trenowania: {e}")
    else:
        st.warning("Najpierw przetwórz dane w etapie 3.")

elif step == "5. Ocena i Eksport":
    st.header("Wyniki i Raport")
    
    if st.session_state.model is not None:
        evaluator = ModelEvaluator()
        task_type = st.session_state.task_type

        metrics = evaluator.calculate_metrics(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test,
            task_type
        )

        st.subheader("Metryki Modelu")
        col1, col2, col3, col4 = st.columns(4)
        keys = list(metrics.keys())

        for i, (k, v) in enumerate(metrics.items()):
            with [col1, col2, col3, col4][i % 4]:
                st.metric(label=k, value=round(v, 4))

        st.subheader("Wizualizacja Wyników")
        if task_type == "Klasyfikacja":
            st.pyplot(evaluator.plot_confusion_matrix(st.session_state.model, st.session_state.X_test, st.session_state.y_test))
        else:
            st.pyplot(evaluator.plot_regression_results(st.session_state.model, st.session_state.X_test, st.session_state.y_test))
            
        st.divider()

        st.subheader("Eksport")

        import joblib
        import io
        
        buffer = io.BytesIO()
        joblib.dump(st.session_state.model, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="Pobierz Wytrenowany Model (.pkl)",
            data=buffer,
            file_name="model.pkl",
            mime="application/octet-stream"
        )

        csv = st.session_state.df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Pobierz Przetworzone Dane (CSV)",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("Najpierw wytrenuj model w etapie 4.")