import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as ticker


data_file_path = 'data_store/modelling/hyper_parameter_tuning/hyper_parameter_tuning_results.csv'


def time_formatter(x, pos):
    hours = int(x)
    minutes = int((x * 60) % 60)
    seconds = int((x * 3600) % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def plot_time_taken_bar(model_id: str, show=True, save_fig=False):
    data = pd.read_csv(data_file_path)
    data = data[data['model_id'] == model_id]
    model_name = data['model_name'].iloc[0]

    for column in ['random_search_time_taken', 'grid_search_time_taken', 'optuna_time_taken']:
        data[column] = pd.to_timedelta(data[column]).dt.total_seconds() / 3600

    new_data = pd.DataFrame(columns=['random_search', 'grid_search', 'optuna'])
    new_data['random_search'] = data['random_search_time_taken']
    new_data['grid_search'] = data['grid_search_time_taken']
    new_data['optuna'] = data['optuna_time_taken']

    ax = new_data.plot(kind='bar', figsize=(6, 4))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title('Time taken for hyperparameter tuning of ' + model_name)
    plt.ylabel('Time taken (hours)')

    plt.xticks([])
    plt.tight_layout()

    if save_fig:
        plt.savefig(f'report_store/modelling/hyper_parameter_tuning/{model_id}_time_taken.png')

    if show:
        plt.show()

    return ax


def plot_metric_bar(model_id: str, metric: str, show=True, save_fig=False):
    data = pd.read_csv(data_file_path)
    data = data[data['model_id'] == model_id]
    model_name = data['model_name'].iloc[0]

    new_data = pd.DataFrame(columns=['random_search', 'grid_search', 'optuna'])
    new_data['random_search'] = data['random_search_cv_' + metric]
    new_data['grid_search'] = data['grid_search_cv_' + metric]
    new_data['optuna'] = data['optuna_cv_' + metric]

    ax = new_data.plot(kind='bar', figsize=(6, 4))

    # Add labels to the top of the bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(),
                '{0:.6f}'.format(p.get_height()),
                fontsize=12, color='black', ha='center', va='bottom')

    if metric == 'r2':
        min_value = new_data.min().min()
        plt.ylim(min_value * 0.99, 1)
    else:
        max_value = new_data.max().max()
        plt.ylim(0, max_value * 1.1)

    plt.title(metric + ' for hyperparameter tuning of ' + model_name)
    plt.ylabel(metric)

    plt.xticks([])
    plt.tight_layout()

    if save_fig:
        plt.savefig(f'report_store/modelling/hyper_parameter_tuning/{model_id}_{metric}_bar.png')

    if show:
        plt.show()

    return ax



