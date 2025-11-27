import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_boxplots(df, numeric_columns):
    n = len(numeric_columns)
    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 5, rows * 4))

    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(y=df[col])
        plt.title(col)

    plt.tight_layout()
    plt.show()
