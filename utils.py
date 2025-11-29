
"""
Utility plotting and helper functions for the Stress Prediction project.

This file originally comes from:
https://github.com/sowinskipatryk/ML-Stress-Prediction

Additional comments and documentation added by:
Sumit Nana Jadhav
"""

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_boxplots(df, numeric_columns, num_cols=3):
    """
    Creates boxplots for numeric columns.
    Helps detect outliers visually.

    Parameters:
        df (DataFrame) - dataset
        numeric_columns (list) - features to plot
        num_cols (int) - number of columns in subplot
    """
    n = len(numeric_columns)
    rows = math.ceil(n / num_cols)
    plt.figure(figsize=(num_cols * 5, rows * 4))

    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(rows, num_cols, i)
        sns.boxplot(y=df[col])
        plt.title(col)

    plt.tight_layout()
    plt.show()


def time_to_float(time_str):
    """
    Converts time format like '10:30 PM' into decimal hours.
    Example: "10:30 PM" -> 22.5

    Useful for converting bedtime/wake time values.
    """
    t = datetime.strptime(time_str, '%I:%M %p').time()
    return t.hour + t.minute / 60.0


def plot_kde_by_stress_level(df, key_features, LABEL_TO_STRESS_LEVEL, num_rows=1, num_cols=3):
    """
    Draws KDE distribution plots for features separated by stress levels.

    Good for comparing how features differ across classes.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

    for i, col in enumerate(key_features):
        for stress_level, label in LABEL_TO_STRESS_LEVEL.items():
            sns.kdeplot(
                df[df['Stress_Detection'] == stress_level][col],
                label=f"Stress: {label}",
                ax=axes[i],
                fill=True,
                alpha=0.5
            )

        axes[i].set_title(f"Distribution of {col}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df_corr):
    """
    Draws a heatmap showing correlation between features.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df_corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        linecolor='black'
    )
    plt.title('Correlation Heatmap')
    plt.show()


def plot_stress_level_class_balance(class_counts, LABEL_TO_STRESS_LEVEL):
    """
    Shows the count of each stress class.
    Useful to detect class imbalance.
    """
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=[LABEL_TO_STRESS_LEVEL[i] for i in class_counts.index],
        y=class_counts.values
    )
    plt.title("Stress Level Class Balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names):
    """
    Draws a confusion matrix to evaluate model predictions.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
