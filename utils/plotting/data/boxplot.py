import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.plotting import config


def plot_boxplot(dataframe: pd.DataFrame, x_var: str):
    x = dataframe[x_var]

    sns.boxplot(x=x, orient='h', whis=10, width=0.25)

    plt.yticks([])
    plt.xlabel(x_var)
    plt.title('Boxplot of the distribution of ' + x_var)

    q1 = round(x.quantile(0.25), 3)
    q3 = round(x.quantile(0.75), 3)
    iqr = round(q3 - q1, 3)

    plt.text(q1, -0.2, f'Q1: {q1}', fontsize=10, horizontalalignment='center', verticalalignment='center')
    plt.text(q3, -0.2, f'Q3: {q3}', fontsize=10, horizontalalignment='center', verticalalignment='center')
    plt.text((q1 + q3) / 2, 0.2, f'IQR: {iqr}', fontsize=10, horizontalalignment='center',
             verticalalignment='center')

    plt.show()


def plot_combined_boxplot(dataframe: pd.DataFrame, remove_originals=False):
    if remove_originals:
        title_placeholder = "new"
        dataframe = dataframe.drop(columns=config.ORIGINAL_FEATURES)
    else:
        if dataframe.columns == config.ORIGINAL_FEATURES:
            title_placeholder = "original"
        else:
            title_placeholder = "all"

        dataframe = dataframe.drop(columns=['Std_dev'])

    sns.boxplot(data=dataframe, orient='h', whis=10, width=0.25, log_scale=True, palette='Blues')

    plt.title(f'Boxplot of the distribution of {title_placeholder} features [log scale]')
    plt.tight_layout()
    plt.show()
