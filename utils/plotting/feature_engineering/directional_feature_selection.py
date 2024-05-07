import matplotlib.pyplot as plt
import numpy as np


def plot_feature_selection_performance_metrics(feature_selection_log):
    model = feature_selection_log[-1]['model']
    feature_selection_log = feature_selection_log[:-1]

    if 'feature_added' in feature_selection_log[0]:
        direction = 'Forward'
    elif 'feature_removed' in feature_selection_log[0]:
        direction = 'Backward'
    else:
        raise ValueError("feature_selection_log must contain either 'feature_added' or 'feature_removed'")

    if direction == 'Forward':
        feature_key = 'feature_added'
    else:
        feature_key = 'feature_removed'

    features = [log[feature_key] for log in feature_selection_log]
    r2 = [log['r2'] for log in feature_selection_log]
    mae = [log['MAE'] for log in feature_selection_log]
    mse = [log['MSE'] for log in feature_selection_log]
    rmse = [log['RMSE'] for log in feature_selection_log]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    fig.suptitle(f'Performance Metrics for {direction} Feature Selection for {model}', fontsize=16)

    for ax in axs.flat:
        ax.xaxis.set_tick_params(rotation=70)
        if direction == 'Forward':
            ax.set_xlabel('Feature Added')
        else:
            ax.set_xlabel('Feature Removed')

    axs[0, 0].plot(features, r2, label='R2', color='blue')
    axs[0, 0].set_title('R2')
    axs[0, 0].set_ylabel('R2 Score')
    r2_std_dev = round(np.std(r2), 4)
    r2_max = round(max(r2), 4)
    axs[0, 0].set_ylim([r2_max - r2_std_dev, 1.01])

    axs[0, 1].plot(features, mae, label='MAE', color='red')
    axs[0, 1].set_title('MAE')
    axs[0, 1].set_ylabel('MAE Score')
    mae_std_dev = round(np.std(mae), 4)
    mae_min = round(min(mae), 4)
    axs[0, 1].set_ylim([0, mae_min + mae_std_dev])

    axs[1, 0].plot(features, mse, label='MSE', color='green')
    axs[1, 0].set_title('MSE')
    axs[1, 0].set_ylabel('MSE Score')
    mse_std_dev = round(np.std(mse), 4)
    mse_min = round(min(mse), 4)
    axs[1, 0].set_ylim([0, mse_min + mse_std_dev])

    axs[1, 1].plot(features, rmse, label='RMSE', color='blue')
    axs[1, 1].set_title('RMSE')
    axs[1, 1].set_ylabel('RMSE Score')
    rmse_std_dev = round(np.std(rmse), 4)
    rmse_min = round(min(rmse), 4)
    axs[1, 1].set_ylim([0, rmse_min + rmse_std_dev])

    plt.tight_layout()
    plt.show()

    return fig
