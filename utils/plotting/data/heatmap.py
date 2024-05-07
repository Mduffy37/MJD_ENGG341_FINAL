import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd


def plot_heatmap(dataframe
                 : pd.DataFrame):
    dataframe = dataframe.drop(columns=['Std_dev'])
    corr = dataframe.corr()

    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')

    # Find the index of the 'Keff' column
    keff_index = list(corr.columns).index('Keff')

    # Add vertical border to the 'Keff' column
    rect_vertical = patches.Rectangle((keff_index, 0), 1, len(corr), linewidth=2,
                                      edgecolor='w', facecolor='none')

    # Add horizontal border to the 'Keff' row
    rect_horizontal = patches.Rectangle((0, keff_index), len(corr), 1, linewidth=2,
                                        edgecolor='w', facecolor='none')

    ax.add_patch(rect_vertical)
    ax.add_patch(rect_horizontal)

    plt.title('Heatmap of the correlation between the features')

    plt.tight_layout()
    plt.show()
