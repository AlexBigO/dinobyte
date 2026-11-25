"""Module containing plot utils"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, **kwargs):
    cm = confusion_matrix(y_true, y_pred)


def plot_data(
    x_data,
    y_data: list,
    yerr_data: list,
    labels_data: list,
    x_model,
    y_model: list,
    labels_model: list,
    xlabel: str,
    ylabel: str,
    figsize: tuple = (8, 10),
):
    fig, ax = plt.subplots(figsize=figsize, dpi=100, tight_layout=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for y, yerr, label in zip(y_data, yerr_data, labels_data):
        ax.errorbar(x_data, y, yerr=yerr, fmt=".", label=label, ecolor="black")
    for y, label in zip(y_model, labels_model):
        ax.plot(x_model, y, label=label)
    ax.legend(loc="best")

    return fig, ax


def plot_history(
    history: dict,
    metrics={"Accuracy": ["accuracy", "val_accuracy"], "Loss": ["loss", "val_loss"]},
    figsize=(8, 6),
):

    for title, curves in metrics.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel("Epoch")
        for c in curves:
            if c == "loss" or c == "mse" or c == "mae":
                color = "darkcyan"
            else:
                color = "darksalmon"
            plt.plot(history[c], color=color)
        plt.legend(curves, loc="best")
        # plt.savefig(f"history_{curves}.pdf")
        plt.show()
