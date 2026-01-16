# Data Science Assistant: AutoML Web App

**Narzędzie do automatyzacji procesu Machine Learningu.**

Aplikacja umożliwia użytkownikowi przeprowadzenie pełnego procesu Data Science: wczytanie surowych danych, eksploracyjną analizę danych i preprocessing, aż po trenowanie i ocenę modeli predykcyjnych.

## Stack

Projekt został zrealizowany w Pythonie, dodatkowo:

* **Frontend/UI:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## Działanie aplikacji

### 1. Obsługa danych
* Wczytywanie plików w formatach: `.csv`, `.xlsx`, `.json`.
* Typy danych rozpoznawane są automatycznie

### 2. Eksploracja Danych
* Generowanie statystyk opisowych (`describe`, braki danych).
* Interaktywne wizualizacje:
    * Histogramy rozkładu zmiennych
    * Heatmapy korelacji
    * Wykresy punktowe scatter plots

### 3. Zaawansowany Preprocessing
Konfigurowalny pipeline czyszczenia danych:
* **Obsługa braków:** Usuwanie, imputacja średnią, medianą albo modą
* **Encoding:** One-Hot Encoding oraz Label Encoding dla zmiennych kategorycznych
* **Skalowanie:** StandardScaler, MinMaxScaler
* **Split:** Decyzja o proporcjach train/test

### 4. Trenowanie modeli
Wsparcie dla problemów **Regresji** i **Klasyfikacji** z wykorzystaniem biblioteki Scikit-learn.
* **Dostępne algorytmy:**
    * Random Forest (Klasyfikator/Regresor)
    * Linear / Logistic Regression
    * Decision Trees
    * SVM, KNN
* **Optymalizacja:** Możliwość włączenia **Grid Search** do automatycznego doboru hiperparametrów (może znacznie wydłużyć czas tworzenia dla dużych zbiorów)

### 5. Ocena i eksport
* Wizualizacja wyników: Macierz pomyłek (Confusion Matrix), wykresy Rzeczywiste vs Przewidywane.
* Metryki: Accuracy, F1-Score, RMSE, MAE, R².
* **Pobieranie:** Możliwość zapisu wytrenowanego modelu (`.pkl`) oraz przetworzonych danych (`.csv`).

---

### Struktura Projektu
Kod został podzielony na moduły dla zachowania czystości:

- `main.py` - Główny plik uruchumieniowy aplikacji
- `src/data_loader.py` - Logika wczytywania plików
- `src/preprocessing.py` - Klasy transformujące dane
- `src/model_builder.py` - Zarządzanie modelami ML
- `src/visualization.py` - Generowanie wykresów
- `src/evaluation.py` - Obliczanie metryk
- `requirements.txt` - Biblioteki

## Instalacja i uruchomienie
1. **Sklonuj repozytorium:**
   ```
   git clone https://github.com/sebbmon/data-science-assistant.git
   ```

2. **Utwórz i aktywuj wirtualne środowisko:**
   ```
   python -m venv venv
   source venv/bin/activate    # linux mac
   venv\Scripts\activate       # windows
   ```

3. **Zainstaluj wymagane pakiety:**
   ```
   pip install -r requirements.txt
   ```

4. **Uruchom serwer:**
   ```
   streamlit run main.py
   ```
