from ..core.utlities import Algo
import os
import matplotlib.pyplot as plt
import hashlib
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score


class Algo3(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH03_classification", to_data_path="mnist_784", target_field="number")
        self.DATA = None
        self.load_openml_data()

    def load_openml_data(self, name='mnist_784', num_training=60000):
        data_ = fetch_openml(name, version=1, cache=True)
        data_.target = data_.target.astype(np.int8)  # fetch_openml() returns targets as

        ind_target_train = sorted([(target, i) for i, target in enumerate(data_.target[:num_training])])
        reorder_train = np.array(ind_target_train)[:, 1]
        ind_target_test = sorted([(target, i) for i, target in enumerate(data_.target[num_training:])])
        reorder_test = np.array(ind_target_test)[:, 1]

        data_.data[:num_training] = data_.data[reorder_train]
        data_.target[:num_training] = data_.target[reorder_train]
        data_.data[num_training:] = data_.data[reorder_test + 60000]
        data_.target[num_training:] = data_.target[reorder_test + 60000]

        self.TRAIN_DATA = data_.data[:num_training]
        self.TRAIN_TARGET = data_.target[:num_training]
        self.TEST_DATA = data_.data[num_training:]
        self.TEST_TARGET = data_.target[num_training:]

    def plot_digits(self, instances, images_per_row=10, **options):
        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row : (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=mpl.cm.binary, **options)
        plt.axis("off")

    def plot_precision_recall_roc(self, sgd_clf, X_train, y_train, fig_id="fig", cv=3):
        y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=cv, method="decision_function")
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        fig_id_1 = fig_id + "_1"
        fig_id_2 = fig_id + "_2"
        fig_id_3 = fig_id + "_3"
        self.plot_precision_recall_threshold(thresholds, precisions, recalls, fig_id_1)
        self.plot_precision_vs_recall(precisions, recalls, fig_id_2)
        self.plot_roc(y_train, y_scores, fig_id_3)
        roc_auc_score_ = roc_auc_score(y_train, y_scores)
        return fig_id_1, fig_id_2, fig_id_3, roc_auc_score_

    def plot_precision_recall_threshold(self, thresholds, precisions, recalls, fig_id):
        plt.figure(figsize=(8, 4))
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.xlabel("Threshold", fontsize=16)
        plt.legend(loc="upper left", fontsize=16)
        plt.ylim([0, 1])
        plt.xlim([-700000, 700000])
        self.save_fig(fig_id)

    def plot_precision_vs_recall(self, precisions, recalls, fig_id):
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, "b-", linewidth=2)
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.axis([0, 1, 0, 1])
        self.save_fig(fig_id)

    def plot_roc(self, y_train, y_scores, fig_id):
        fpr, tpr, thresholds = roc_curve(y_train, y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=None)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        self.save_fig(fig_id)

    def plot_comparison_roc(self, algs, X_train, y_train, fig_id="fig", cv=3):
        plt.figure(figsize=(8, 6))
        n=0
        for alg in algs:
            y_scores = cross_val_predict(algs[alg]['alg'], X_train, y_train, cv=cv, method=algs[alg]['method'])
            try:
                if y_scores.shape[1]:
                    y_scores = y_scores[:, 1]
            except Exception as e:
                pass
                # print(e)
            algs[alg]['y_scores'] = y_scores
            if algs[alg]['eval'] != '':
                print('alg_21' + alg)
                eval(algs[alg]['eval'])
                print('alg_22' + alg)
            fpr, tpr, thresholds = roc_curve(y_train, y_scores)
            algs[alg]['fpr'], algs[alg]['tpr'], algs[alg]['thresholds'] = fpr, tpr, thresholds

            roc_auc_score_ = roc_auc_score(y_train, y_scores)
            algs[alg]['roc_auc_score_'] = roc_auc_score_
            if n == 0:
                plt.plot(algs[alg]['fpr'], algs[alg]['tpr'], "b:", linewidth=2, label=alg + "(auc=" + str(round(100*roc_auc_score_))+"%)")
            else:
                plt.plot(algs[alg]['fpr'], algs[alg]['tpr'], linewidth=2, label=alg + "(auc=" + str(round(100*roc_auc_score_))+"%)")
            n += 1
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right", fontsize=16)
        self.save_fig(fig_id)
        return algs

    def plot_confusion_matrix_gray(self, matrix, fig_id="confusion_matrix_plot"):
        plt.figure(figsize=(8,8))
        plt.matshow(matrix, cmap=plt.cm.gray)
        self.save_fig(fig_id, tight_layout=False)

    def plot_confusion_matrix_color(self, matrix, fig_id="confusion_matrix_plot"):
        """If you prefer color and a colorbar"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        self.save_fig(fig_id, tight_layout=False)


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
