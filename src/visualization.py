import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Rozkład zmiennej: {column}")
    return fig

def plot_scatter(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{x_col} vs {y_col}")
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return fig
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Macierz Korelacji")
    return fig