from .utlities import Algo
import os
import matplotlib.pyplot as plt
import hashlib
import numpy as np
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


class AlgoG(Algo):
    def __init__(self):
        super().__init__(chapter_id="gold", to_data_path="gold", target_field=None)
        self.load_excel_data("best_clients_bronzallure_com")