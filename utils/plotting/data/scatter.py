import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import config


def plot_scatter(dataframe: pd.DataFrame, x_var: str, y_var: str):
    x = dataframe[x_var]
    y = dataframe[y_var]

    sns.scatterplot(x=x, y=y, color='blue', alpha=0.5)

    plt.title('Scatterplot of ' + x_var + ' vs ' + y_var)
    plt.show()


def plot_combined_scatter(dataframe: pd.DataFrame, target: str, remove_originals=False):
    plot_columns = [col for col in dataframe.columns if col != target]
    plot_columns.remove('Std_dev')

    if remove_originals:
        plot_columns = [col for col in plot_columns if col not in config.ORIGINAL_FEATURES]

    num_plots = len(plot_columns)

    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    fig = plt.figure(figsize=(15, 15))

    for i, column in enumerate(plot_columns, start=1):
        ax = fig.add_subplot(num_rows, num_cols, i)
        sns.scatterplot(x=target, y=column, data=dataframe, color="skyblue", alpha=0.5, ax=ax)
        ax.set_title('Scatterplot of ' + target + ' vs ' + column)

    plt.tight_layout()
    plt.show()