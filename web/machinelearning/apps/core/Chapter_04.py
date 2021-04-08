from .utlities import Algo
import os
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class Algo4(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH04_training_models", target_field=None)
        self.TRAIN = 2 * np.random.rand(100, 1)
        self.TRAIN_TARGET = 4 + 3 * self.TRAIN + np.random.randn(100, 1)
        self.X_b = np.c_[np.ones((100, 1)), self.TRAIN]  # add x0 = 1 to each instance

        self.X_new = np.array([[0], [2]])
        self.X_new_b = np.c_[np.ones((2, 1)), self.X_new]  # add x0 = 1 to each instance

    def show_effect_of_learning_rate(self):
        theta = np.random.randn(2, 1)  # random initialization
        plt.figure(figsize=(10, 4))
        plt.subplot(131)
        self.plot_gradient_descent(theta, eta=0.02)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.subplot(132)
        self.plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
        plt.subplot(133)
        self.plot_gradient_descent(theta, eta=0.5)
        fig_id = "gradient_descent_plot"
        self.save_fig(fig_id)
        return fig_id

    def plot_gradient_descent(self, theta, eta, theta_path=None):
        m = len(self.X_b)
        plt.plot(self.TRAIN, self.TRAIN_TARGET, "b.")
        n_iterations = 1000
        for iteration in range(n_iterations):
            if iteration < 10:
                y_predict = self.X_new_b.dot(theta)
                style = "b-" if iteration > 0 else "r--"
                plt.plot(self.X_new, y_predict, style)
            gradients = 2 / m * self.X_b.T.dot(self.X_b.dot(theta) - self.TRAIN_TARGET)
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 2, 0, 15])
        plt.title(r"$\eta = {}$".format(eta), fontsize=16)