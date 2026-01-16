from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
                            mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    
    def calculate_metrics(self, model, X_test, y_test, task_type):
        y_pred = model.predict(X_test)
        
        if task_type == "Klasyfikacja":
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (Macro)": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "Recall (Macro)": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "F1 Score (Macro)": f1_score(y_test, y_pred, average='macro', zero_division=0)
            }
        else:
            return {
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2 Score": r2_score(y_test, y_pred)
            }

    def plot_confusion_matrix(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Macierz pomyłek (Confusion Matrix)")
        ax.set_ylabel("Rzeczywiste")
        ax.set_xlabel("Przewidywane")
        return fig

    def plot_regression_results(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Wartości Rzeczywiste")
        ax.set_ylabel("Wartości Przewidywane")
        ax.set_title("Rzeczywiste vs Przewidywane")
        return fig