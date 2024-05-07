import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_path = 'data_store/modelling/model_performance/model_evaluation_results.csv'
results = pd.read_csv(results_path)


def plot_all_models_test_results(metric: str, remove_bad_models: bool = True):

    results_to_plot = results[['model_id', 'model_name', f'cv_{metric}', f'test_{metric}']]

    if remove_bad_models:
        if metric == 'r2':
            results_to_plot = results_to_plot[results_to_plot[f'test_{metric}'] > 0.99]
        elif metric == 'mae':
            results_to_plot = results_to_plot[results_to_plot[f'test_{metric}'] < 0.01]
        elif metric == 'mse':
            results_to_plot = results_to_plot[results_to_plot[f'test_{metric}'] < 0.0002]
        elif metric == 'rmse':
            results_to_plot = results_to_plot[results_to_plot[f'test_{metric}'] < 0.015]

    if metric == 'r2':
        results_to_plot = results_to_plot.sort_values(by=f'test_{metric}', ascending=False)
    else:
        results_to_plot = results_to_plot.sort_values(by=f'test_{metric}', ascending=True)

    model_names = results_to_plot['model_id']
    cv_scores = results_to_plot[f'cv_{metric}']
    test_scores = results_to_plot[f'test_{metric}']

    # Set up positions for bars
    x = np.arange(len(model_names))
    bar_width = 0.35

    # Plotting
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - bar_width / 2, cv_scores, bar_width, label=f'CV {metric}')
    bars2 = ax.bar(x + bar_width / 2, test_scores, bar_width, label=f'Test {metric}')

    # Labeling
    ax.set_xlabel('Model')
    ax.set_ylabel(f'{metric} Score')
    ax.set_title(f'CV {metric} and Test {metric} by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    if metric == 'r2':
        min_val = min(min(cv_scores), min(test_scores))
        ax.set_ylim(min_val * 0.95, 1.0)
    else:
        max_val = max(max(cv_scores), max(test_scores))
        ax.set_ylim(0, max_val * 1.1)

    # Show plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()