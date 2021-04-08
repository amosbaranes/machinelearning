from .utlities import Algo
import tarfile
import os
from fancyimpute import IterativeImputer
from six.moves import urllib
import urllib.request
import tarfile
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd

from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook, load_workbook

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


class AlgoP1(Algo):
    def __init__(self):
        super().__init__(chapter_id="PROJ01_dual_single", to_data_path="dual_single", target_field="EPSI")
        file_from_ = "dual_single.csv"
        file_to_ = "dual_single.csv.gz"
        # get data ---
        csv_path = os.path.join(self.TO_DATA_PATH, file_from_)
        if not os.path.isfile(csv_path):
            self.DUAL_SINGLE_URL = "https://github.com/amosbaranes/ml_data/raw/master/dual_single.csv.gz"
            self.fetch_tgz_data(self.DUAL_SINGLE_URL, file_from_, "gz")
        self.load_csv_data("dual_single")
        self.DATA_SOURCE = self.DATA.iloc[:, 5:19]
        self.DATA_IMPUTED = self.DATA.iloc[:, 19:]

        self.SINGLE_DATA = self.DATA[self.DATA["Type"] == "Single"]
        self.DUAL_DATA = self.DATA[self.DATA["Type"] == "Dual"]
        self.SINGLE_DATA_IMPUTED = self.SINGLE_DATA.iloc[:, 19:]
        self.DUAL_DATA_IMPUTED = self.DUAL_DATA.iloc[:, 19:]

        self.SINGLE_DATA_SOURCE = self.SINGLE_DATA.iloc[:, 5:19]
        self.DUAL_DATA_SOURCE = self.DUAL_DATA.iloc[:, 5:19]
        self.SINGLE_DATA_SOURCE_ZZ = zig_zag_(self.SINGLE_DATA_SOURCE, a_rows=0.25, a_col=0.85)
        self.DUAL_DATA_SOURCE_ZZ = zig_zag_(self.DUAL_DATA_SOURCE, a_rows=0.25, a_col=0.85)

        mice_impute_s = IterativeImputer()
        self.SINGLE_DATA_SOURCE_ZZI = pd.DataFrame(mice_impute_s.fit_transform(self.SINGLE_DATA_SOURCE_ZZ))
        # print(self.SINGLE_DATA_SOURCE_ZZ.columns)
        # print(self.SINGLE_DATA_SOURCE_ZZI.columns)
        try:
            self.SINGLE_DATA_SOURCE_ZZI.columns = self.SINGLE_DATA_SOURCE_ZZ.columns
            file = "SINGLE_DATA_IMPUTED"
            ssr = os.path.join(self.TO_DATA_PATH, file + ".xlsx")  # "housing"
            print(ssr)
            with pd.ExcelWriter(ssr, engine='xlsxwriter') as writer:
                self.SINGLE_DATA_SOURCE_ZZI.to_excel(writer, sheet_name="imputed")
                writer.save()
        except Exception as e:
            print(e)
        # print(self.SINGLE_DATA_SOURCE_ZZI)

        mice_impute_d = IterativeImputer()
        self.DUAL_DATA_SOURCE_ZZI = pd.DataFrame(mice_impute_d.fit_transform(self.DUAL_DATA_SOURCE_ZZ))
        try:
            self.DUAL_DATA_SOURCE_ZZI.columns = self.DUAL_DATA_SOURCE_ZZ.columns
            file = "DUAL_DATA_IMPUTED"
            ssr = os.path.join(self.TO_DATA_PATH, file + ".xlsx")  # "housing"
            print(ssr)
            with pd.ExcelWriter(ssr, engine='xlsxwriter') as writer:
                self.DUAL_DATA_SOURCE_ZZI.to_excel(writer, sheet_name="imputed")
                writer.save()
        except Exception as e:
            print(e)
        # print(self.DUAL_DATA_SOURCE_ZZI)


def zig_zag_(df, a_rows=0.20, a_col=0.9):
    # print(df.shape)
    n_pc = int(round((1 - a_rows) * df.shape[1], 0))
    # print(df.shape[1], n_pc)

    df = df.copy()
    for n in range(1, n_pc):
        df = df.dropna(thresh=n)
        # print('n', n, df.shape)
        for k in df.columns:
            npk = df.isnull().sum()[k] / df.shape[0]
            cpk = (1 - n / df.shape[1])
            # print('k', k, npk, cpk)
            if npk > cpk:
                try:
                    df = df.drop(k, axis=1)
                except Exception as ex:
                    print(ex)
                # print(df.shape)
    return df
