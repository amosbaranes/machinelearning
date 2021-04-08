from .utlities import Algo
import os
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd
import multiprocessing
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
from os import listdir
from ..introml.models import (Accounts, Companies, CompanyPeriodAccountValue, CompanyPeriodAccountGeneral, Periods)


class Algo2(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH02_end_to_end_project", to_data_path="housing", target_field="median_house_value")
        csv_path = os.path.join(self.TO_DATA_PATH, "housing.csv")
        if not os.path.isfile(csv_path):
            self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
            self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
            self.fetch_tgz_data(self.HOUSING_URL, "housing")
        self.load_csv_data("housing")

        xlsx_path = self.TO_DATA_PATH + "/SEC_FINANCIAL_STATEMENTS/2012v.xlsx"
        if not os.path.isfile(xlsx_path):
            # Accounts.truncate()
            # Companies.truncate()
            print('downloading')
            zip_url = "https://sites.google.com/a/drbaranes.com/ac_ml/home/SEC_FINANCIAL_STATEMENTS.zip?attredirects=0&d=1"
            self.fetch_zip_data(zip_url, 'sec')
            # files = [f for f in listdir(self.TO_DATA_PATH) if isfile(join(self.TO_DATA_PATH, f))]

            df_fields = self.load_excel_data(file='SEC_FINANCIAL_STATEMENTS/Fields', sheet_name='accounts')
            # print(df_fields)
            for index, row in df_fields.iterrows():
                category_ = row['Category']
                fields_ = row['Field']
                account_name_ = fields_[1:len(fields_)-1]
                # print(fields_, account_name_)
                account, created = Accounts.objects.get_or_create(category=category_, account=fields_,
                                                                  account_name=account_name_)

            df_fields = self.load_excel_data(file='SEC_FINANCIAL_STATEMENTS/Fields', sheet_name='cik')
            for index, row in df_fields.iterrows():
                cik_ = row['CIK']
                # print(cik_)
                account, created = Companies.objects.get_or_create(cik=cik_)

        print('='*20)
        print('End Download')
        print('='*20)

        file_years_quarter = [(year, 0) for year in range(2020, 2021)]
        for nq in file_years_quarter:
            self.upload_data(nq)
            print('=' * 20)
            print('End process', nq)
            print('=' * 20)

        # with multiprocessing.Pool() as pool:
        #     pool.map(self.upload_data, file_years_quarter)

    def upload_data(self, nq):
        year = nq[0]
        quarter = nq[1]
        # print(year, quarter)

        try:
            year_quarter_ = int(year*100 + quarter)
            print(year_quarter_)
            period_, created = Periods.objects.get_or_create(year_quarter=year_quarter_, year=year, quarter=quarter)
            print(period_)
        except Exception as exc:
            print(exc)

        df_fields = self.load_excel_data(file='SEC_FINANCIAL_STATEMENTS/Fields', sheet_name='accounts')

        df_fields_general = df_fields.loc[df_fields['Table'] == 'CompanyPeriodAccountGeneral']
        df_fields_value = df_fields.loc[df_fields['Table'] == 'CompanyPeriodAccountValue']

        if quarter == 0:
            file = 'SEC_FINANCIAL_STATEMENTS/'+str(year)+'v'
        else:
            file = 'SEC_FINANCIAL_STATEMENTS/'+str(year)+str(quarter)+'v'

        # print(file)
        print(year)
        df = self.load_excel_data(file)
        n__ = 0
        for index, drow in df.iterrows():
            cik_ = str(drow['[CIK]'])
            if n__ % 100 == 0:
                print(n__)
            n__ += 1
            print(n__, cik_)
            company_ = Companies.objects.get(cik=cik_)
            for index, grow in df_fields_general.iterrows():
                account_ = Accounts.objects.get(account=grow['Field'])
                vg_ = drow[grow['Field']]
                if str(vg_) != 'nan' and str(vg_) != 'Company not found':
                    # print('post', company_, account_, vg_, year_quarter_)
                    cpag,created = CompanyPeriodAccountGeneral.objects.get_or_create(company=company_,
                                                                                     account=account_,
                                                                                     value=vg_,
                                                                                     period=period_)
            # print('=value' * 30)
            # print('value', year, cik_, n__)

            for vindex, vrow in df_fields_value.iterrows():
                vaccount_ = Accounts.objects.get(account=vrow['Field'])
                vvg_ = drow[vrow['Field']]
                if str(vvg_) != 'nan' and str(vvg_) != 'Company not found':
                    vcpag,created = CompanyPeriodAccountValue.objects.get_or_create(company=company_,
                                                                                    account=vaccount_,
                                                                                    value=vvg_,
                                                                                    period=period_)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True, algo=None): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.ALGO = algo
        if algo:
            self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX = [
                list(self.ALGO.TRAIN_DATA.columns).index(col)
                for col in ("total_rooms", "total_bedrooms", "population", "households")]
        else:
            self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX = [3, 4, 5, 6]

        # print('CombinedAttributesAdder: the indexes of the fields:')
        # print(self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX)
        # print('-'*100)

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.ROOMS_IX] / X[:, self.HOUSEHOLD_IX]
        population_per_household = X[:, self.POPULATION_IX] / X[:, self.HOUSEHOLD_IX]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.BEDROOMS_IX] / X[:, self.ROOMS_IX]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

