import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class Preprocessor:
    def __init__(self, target_col):
        self.target_col = target_col

    def handle_missing_values(self, df, strategy="Brak"):
        # obsluga brakow danych 4 mozliwosci
        df_clean = df.copy()
        
        if strategy == "Usuń wiersze":
            df_clean = df_clean.dropna()
        elif strategy == "Średnia":
            num_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        elif strategy == "Mediana":
            num_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        elif strategy == "Moda":
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
        return df_clean

    def encode_categorical(self, df, method="Label Encoding"):
        df_encoded = df.copy()
        
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        cat_cols = [c for c in cat_cols if c != self.target_col]

        if method == "Label Encoding":
            le = LabelEncoder()
            for col in cat_cols:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        elif method == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
            
        # obsluga targetu
        if df_encoded[self.target_col].dtype == 'object':
            le_target = LabelEncoder()
            df_encoded[self.target_col] = le_target.fit_transform(df_encoded[self.target_col])
            
        return df_encoded

    def scale_features(self, df, method="StandardScaler"):
        df_scaled = df.copy()
        features = [col for col in df_scaled.columns if col != self.target_col]
        
        if method == "StandardScaler":
            scaler = StandardScaler()
            df_scaled[features] = scaler.fit_transform(df_scaled[features])
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
            df_scaled[features] = scaler.fit_transform(df_scaled[features])
            
        return df_scaled

    def split_data(self, df, test_size=0.2):
        # dzielenie danych na x y i zbiory treningowe testowe
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)