import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import config


def plot_kde(dataframe: pd.DataFrame, x_var: str, y_var: str):
    x = dataframe[x_var]
    y = dataframe[y_var]

    sns.kdeplot(x=x, y=y, fill=True)

    plt.title('Density plot of ' + x_var + ' vs ' + y_var)
    plt.show()


def plot_combined_kde(dataframe: pd.DataFrame, target: str, remove_originals=False):
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
        sns.kdeplot(x=target, y=column, data=dataframe, fill=True, ax=ax, cut=0)
        ax.set_title('Density plot of ' + target + ' vs ' + column)

    plt.tight_layout()
    plt.show()