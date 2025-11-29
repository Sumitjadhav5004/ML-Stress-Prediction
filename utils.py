from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_boxplots(df, numeric_columns, num_cols=3):
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
    time_object = datetime.strptime(time_str, '%I:%M %p').time()

    hours = time_object.hour
    minutes = time_object.minute
    
    decimal_hours = hours + (minutes / 60.0)
    
    return decimal_hours


def plot_kde_by_stress_level(df, key_features, LABEL_TO_STRESS_LEVEL, num_rows=1, num_cols=3):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

    for i, col in enumerate(key_features):
        for stress_level, label in LABEL_TO_STRESS_LEVEL.items():
            sns.kdeplot(
                df[df['Stress_Detection'] == stress_level][col], 
                label=f'Stress: {label}', 
                ax=axes[i], 
                fill=True,
                alpha=0.5
            )
        axes[i].set_title(f'Distribution of {col} by Stress Level')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        linecolor='black'
    )
    plt.title('Correlation Heatmap of Numerical Features and Target')
    plt.show()


def plot_stress_level_class_balance(class_counts, LABEL_TO_STRESS_LEVEL):
    plt.figure(figsize=(6,4))
    sns.barplot(
        x=class_counts.index.map(lambda i: LABEL_TO_STRESS_LEVEL[i]),
        y=class_counts.values
    )
    plt.title("Stress level class balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
