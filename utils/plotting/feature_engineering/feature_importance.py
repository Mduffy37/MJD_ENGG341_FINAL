import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from utils.plotting import config


def plot_feature_importances(info_dict: dict, title: str = None):

    df = pd.DataFrame(list(info_dict['feature_importances'].items()), columns=['Feature', 'Importance'])
    df_sorted = df.sort_values(by='Importance', ascending=False)
    df_sorted.set_index('Feature', inplace=True)

    df_sorted['Importance'] = df_sorted['Importance'] + 1e-10

    metrics = info_dict['metrics']

    if title is None:
        title = 'Feature importances'

    # Create a bar plot of the feature importances
    plt.figure(figsize=(10, 10))
    sns.barplot(x=df_sorted['Importance'], y=df_sorted.index, color=config.COLOURS['blue'])
    plt.ylabel("Features")
    plt.xscale('log')
    plt.title(title)

    # Create a custom legend
    legend_patches = [patches.Patch(color='none',
                                    label=f'{metric}: {round(value, 6)}') for metric, value in metrics.items()]

    plt.legend(handles=legend_patches, frameon=True, fontsize=15, loc='lower right')

    plt.show()


def plot_permutation_importances(permutation_importances: pd.DataFrame):
    pass