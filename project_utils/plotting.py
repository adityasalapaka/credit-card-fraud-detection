# project_utils/plotting.py

from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def smooth_curve(values: Sequence[float], window_size: int = 3) -> np.ndarray:
    """Applies a simple moving average."""
    values_array = np.asarray(values, dtype=float)
    if window_size < 2:
        return values_array
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(values_array, kernel, mode="same")


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    title: str = "Training vs Validation Loss",
    smoothing: int = 1,
) -> None:
    """
    Plots training and validation loss curves, optionally smoothed.

    Args:
        train_losses (list or array): Loss recorded at each training epoch.
        val_losses (list or array): Loss recorded at each validation epoch.
        title (str): Title for the plot.
        smoothing (int): Window size for moving average smoothing.
                         1 = no smoothing.

    Returns:
        None — displays a matplotlib figure.
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 4))

    epochs = range(1, len(train_losses) + 1)

    # Smooth losses
    smoothed_train = smooth_curve(train_losses, smoothing)
    smoothed_val = smooth_curve(val_losses, smoothing)

    # Plot curves
    plt.plot(epochs, smoothed_train, label="Train Loss")
    plt.plot(epochs, smoothed_val, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
