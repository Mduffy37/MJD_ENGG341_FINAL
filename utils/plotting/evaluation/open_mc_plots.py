import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats




def plot_pred_vs_actual(y_pred, y_test):
    low_lim = 0.3
    up_lim = 1.7
    tick_spacing = 0.1

    plt.figure(figsize=(6, 6))
    plt.title(f"Predictions vs OpenMC Values")
    plt.xlabel("OpenMC Keff")
    plt.ylabel("Predicted Keff")
    plt.scatter(y_test, y_pred, s=1, color='#86bdf4')
    plt.xlim([low_lim, up_lim])
    plt.ylim([low_lim, up_lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()

    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], '--', color='#160a67', linewidth=0.75)

    plt.grid(True)

    ticks = np.arange(low_lim, up_lim, tick_spacing)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.tight_layout()
    plt.show()


def plot_residuals_histogram(y_pred, y_test):
    plt.figure(figsize=(6, 6))
    plt.title(f"Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.xlim(-0.02, 0.02)
    plt.hist(y_test - y_pred, bins=75, color='#86bdf4')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_predictions(y_pred, y_test):
    plt.figure(figsize=(6, 6))
    plt.title(f"Residuals vs Predictions")
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, s=1, color='#86bdf4')
    plt.plot(plt.gca().get_xlim(), [0, 0], '--', lw=1, color='#160a67')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

