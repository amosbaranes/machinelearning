from django.shortcuts import render
# ---------------------------------
from ..core.Chapter_01 import Algo1
from ..core.Chapter_02 import Algo2, CombinedAttributesAdder
from ..core.Chapter_03 import Algo3, Never5Classifier
from ..core.Chapter_04 import Algo4
from ..core.PROJ_01 import AlgoP1
from ..core.PROJ_02 import AlgoP2
from ..core.models import Security, SecurityGroup
from ..core.gold import AlgoG
from ..core import stat_utilities as su
# ---------------------------------
from sklearn import linear_model, neighbors
from sklearn.linear_model import LinearRegression

# ---------------------------------
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn import pipeline
from scipy.stats import randint
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from datetime import timedelta, datetime
from pandas_datareader import data as pdr

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy import stats
# ---------------------------------

def index(request):
    return render(request, 'introml/index.html', {'x': 500})


# Main Navigator ---
def show_content(request):
    chapter = request.POST.get('chapter')
    # print(chapter)
    page = request.POST.get('page')
    # print(page)
    # the variable chapters send me to the right chapter function --
    # the variable page send me to the right section in the function function (see below) --
    # --- chapter 1 --
    if str(chapter) == '1':
        return chapter_1(request, page)
    # if str(chapter) == '1_1':
    #     return chapter_1_1(request, page)
    # --- chapter 2 ------
    # --------------------

    elif str(chapter) == '2':
        return chapter_2(request, page)
    elif str(chapter) == '2_1':
        return chapter_2_1(request, page)
    elif str(chapter) == '2_2':
        return chapter_2_2(request, page)
    # --- chapter 3 -----
    # --------------------
    if str(chapter) == '3':
        return chapter_3(request, page)

    # --- chapter 4 -----
    # --------------------
    if str(chapter) == '4':
        return chapter_4(request, page)

    # --- project 1 -----
    # --------------------
    elif str(chapter) == 'p1':
        return project_1(request, page)
    elif str(chapter) == 'p11':
        return project_1_1(request, page)

    # --- project 2 -----
    # --------------------
    elif str(chapter) == 'p2':
        return project_2(request, page)

    # ------- Gold --------
    elif str(chapter) == 'gold':
        return gold_1(request, page)


#  -- Every Chapter has it own code --
# -------------------------------------------------------------------
def ch01(request):
    title = 'The Machine Learning Landscape'
    return render(request, 'introml/ch01.html', {'title': title})


def chapter_1(request, page):
    print('Inside the chapter_1 function')
    if page == "plot_gdp_pc_vs_life_satisfaction":
        print('Inside the page: plot_gdp_pc_vs_life_satisfaction')
        algo = Algo1()
        title = 'GDP/C vs Life satisfaction'

        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

        fig_id = "plot_gdp_pc_vs_life_satisfaction"
        algo.save_fig(fig_id)

        render_ = render(request, 'introml/_show_plot.html',
                         {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})
        del algo
        return render_
    elif page == "linear_model":
        title = 'Select a linear model for Prediction'
        algo = Algo1()
        X = np.c_[algo.sample_data["GDP per capita"]]
        y = np.c_[algo.sample_data["Life satisfaction"]]
        model = linear_model.LinearRegression()  # linear reg
        # model = neighbors.KNeighborsRegressor(n_neighbors=3)   #  k-neighbors
        # Train the model
        model.fit(X, y)
        # Make a prediction for Cyprus
        X_new = [[1]]  # Cyprus' GDP per capita
        print('-'*10)
        print(title)
        print('-'*10)
        print('X_new = ' + str(X_new))
        print('-'*10)
        y = model.predict(X_new)
        print('output:  y = ' + str(y))  # outputs [[ 5.96242338]]
        print('-'*10)
        render_ = render(request, 'introml/_data_processed_successfully.html', {'title': title})
        del algo
        return render_
    elif page == "life_satisfaction_several_countries_on_plot":
        algo = Algo1()
        print(1000)
        title = 'Life Satisfaction Several countries on plot'
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5, 3))
        plt.axis([0, 60000, 0, 10])
        print(10001)
        position_text = {
            "Hungary": (5000, 1),
            "Korea": (18000, 1.7),
            "France": (29000, 2.4),
            "Australia": (40000, 3.0),
            "United States": (52000, 3.8),
        }
        print(10002)
        for country, pos_text in position_text.items():
            print(country, pos_text)
            pos_data_x, pos_data_y = algo.sample_data.loc[country]
            country = "U.S." if country == "United States" else country
            plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                         arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
            plt.plot(pos_data_x, pos_data_y, "ro")
        print(10003)
        fig_id = "money_happy_scatterplot"
        algo.save_fig(fig_id)
        # plt.show()
        saved_data = "lifesat.csv"
        # print(10004)
        # algo.sample_data.to_csv(os.path.join(algo.TO_DATA_PATH, saved_data))
        # print('sample_data')
        # print(10005)
        # print(algo.sample_data.loc[list(position_text.keys())])
        # print(10006)
        render_ = render(request, 'introml/ch01/_several_countries_on_plot.html',
                         {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID,
                          'img_type': 'png', 'saved_data': saved_data})
        del algo
        return render_
    elif page == "tweaking_model_params_plot":
        title = 'GDP/C vs Life satisfaction (diff params)'
        algo = Algo1()
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
        plt.axis([0, 60000, 0, 10])
        X = np.linspace(0, 60000, 1000)
        plt.plot(X, 2 * X / 100000, "r")
        plt.text(40000, 2.7, r"$\theta_0 = 0$", fontsize=14, color="r")
        plt.text(40000, 1.8, r"$\theta_1 = 2 \times 10^{-5}$", fontsize=14, color="r")
        plt.plot(X, 8 - 5 * X / 100000, "g")
        plt.text(5000, 9.1, r"$\theta_0 = 8$", fontsize=14, color="g")
        plt.text(5000, 8.2, r"$\theta_1 = -5 \times 10^{-5}$", fontsize=14, color="g")
        plt.plot(X, 4 + 5 * X / 100000, "b")
        plt.text(5000, 3.5, r"$\theta_0 = 4$", fontsize=14, color="b")
        plt.text(5000, 2.6, r"$\theta_1 = 5 \times 10^{-5}$", fontsize=14, color="b")
        fig_id1 = 'tweaking_model_params_plot'
        fig_name_id1 = 'Tweaking model params plot'
        algo.save_fig(fig_id1)
        # plt.show()
        lin1 = linear_model.LinearRegression()
        Xsample = np.c_[algo.sample_data["GDP per capita"]]
        ysample = np.c_[algo.sample_data["Life satisfaction"]]
        lin1.fit(Xsample, ysample)
        t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]

        print('-'*20)
        print(t0, t1)
        print('-'*20)

        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))

        plt.axis([0, 60000, 0, 10])
        X = np.linspace(0, 60000, 1000)

        plt.plot(X, t0 + t1*X, "b")

        fig_id2 = 'regression_line'
        algo.save_fig(fig_id2)
        fig_name_id2 = 'Regression Line'
        return render(request, 'introml/ch01/_tweaking_model_params_plot.html',
                      {'title': title,
                       'fig_id1': fig_id1, 'fig_name_id1': fig_name_id1,
                       'fig_id2': fig_id2, 'fig_name_id2': fig_name_id2,
                       'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})
    elif page == "nonrepresentative_training_data":
        algo = Algo1()
        title = "Nonrepresentative Training Data"
        position_text2 = {
            "Brazil": (1000, 9.0),
            "Mexico": (11000, 9.0),
            "Chile": (25000, 9.0),
            "Czech Republic": (35000, 9.0),
            "Norway": (60000, 3),
            "Switzerland": (72000, 3.0),
            "Luxembourg": (90000, 3.0),
        }
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
        plt.axis([0, 110000, 0, 10])

        for country, pos_text in position_text2.items():
            pos_data_x, pos_data_y = algo.missing_data.loc[country]
            plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                    arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
            plt.plot(pos_data_x, pos_data_y, "rs")
        # -- Same as the code before --
        lin1 = linear_model.LinearRegression()
        Xsample = np.c_[algo.sample_data["GDP per capita"]]
        ysample = np.c_[algo.sample_data["Life satisfaction"]]
        lin1.fit(Xsample, ysample)
        t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
        print('-'*20)
        print('-'*20)
        print(t0, t1)
        print('-'*20)
        print('-'*20)
        X = np.linspace(0, 110000, 1000)
        plt.plot(X, t0 + t1*X, "b:")
        # --
        lin_reg_full = linear_model.LinearRegression()
        Xfull = np.c_[algo.full_country_stats["GDP per capita"]]
        yfull = np.c_[algo.full_country_stats["Life satisfaction"]]
        lin_reg_full.fit(Xfull, yfull)
        t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]

        print('-'*20)
        print('-'*20)
        print(t0full, t1full)
        print('-'*20)
        print('-'*20)

        X = np.linspace(0, 110000, 1000)
        plt.plot(X, t0full + t1full * X, "k")
        # --
        fig_id = 'representative_training_data_scatterplot'
        algo.save_fig(fig_id)
        # plt.show()
        return render(request, 'introml/ch01/_nonrepresentative_training_data.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png'})
    elif page == 'overfitting_the_training_data':
        algo = Algo1()
        title = 'Overfitting the Training Data'
        algo.full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
        plt.axis([0, 110000, 0, 10])
        # --
        poly = preprocessing.PolynomialFeatures(degree=60, include_bias=False)
        scaler = preprocessing.StandardScaler()
        lin_reg2 = linear_model.LinearRegression()
        pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
        Xfull = np.c_[algo.full_country_stats["GDP per capita"]]
        yfull = np.c_[algo.full_country_stats["Life satisfaction"]]
        pipeline_reg.fit(Xfull, yfull)

        X = np.linspace(0, 110000, 1000)
        curve = pipeline_reg.predict(X[:, np.newaxis])
        plt.plot(X, curve)
        fig_id = 'overfitting_model_plot'
        algo.save_fig(fig_id)
        # plt.show()

        return render(request, 'introml/ch01/_overfitting_the_training_data.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png'})
    elif page == 'regularization':
        algo = Algo1()
        title = 'Regularization'
        plt.figure(figsize=(8,3))
        plt.xlabel("GDP per capita")
        plt.ylabel('Life satisfaction')

        plt.plot(list(algo.sample_data["GDP per capita"]), list(algo.sample_data["Life satisfaction"]), "bo")
        plt.plot(list(algo.missing_data["GDP per capita"]), list(algo.missing_data["Life satisfaction"]), "rs")
        # --
        lin_reg_full = linear_model.LinearRegression()
        Xfull = np.c_[algo.full_country_stats["GDP per capita"]]
        yfull = np.c_[algo.full_country_stats["Life satisfaction"]]
        lin_reg_full.fit(Xfull, yfull)
        t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
        X = np.linspace(0, 110000, 1000)
        plt.plot(X, t0full + t1full * X, "r--", label="Linear model on all data")
        # --
        lin1 = linear_model.LinearRegression()
        Xsample = np.c_[algo.sample_data["GDP per capita"]]
        ysample = np.c_[algo.sample_data["Life satisfaction"]]
        lin1.fit(Xsample, ysample)
        t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
        X = np.linspace(0, 110000, 1000)
        plt.plot(X, t0 + t1 * X, "b:", label="Linear model on partial data")
        # --
        ridge = linear_model.Ridge(alpha=10 ** 9.5)
        Xsample = np.c_[algo.sample_data["GDP per capita"]]
        ysample = np.c_[algo.sample_data["Life satisfaction"]]
        ridge.fit(Xsample, ysample)
        t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
        plt.plot(X, t0ridge + t1ridge * X, "b", label="Regularized linear model on partial data")
        # --
        plt.legend(loc="lower right")
        plt.axis([0, 110000, 0, 10])
        fig_id = 'ridge_model_plot'
        algo.save_fig(fig_id)
        # plt.show()
        return render(request, 'introml/ch01/_regularization.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png'})
    elif page == 'random_charts':
        algo = Algo1()
        title = 'Random charts example'
        plt.figure(figsize=(8, 3))

        ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
        ts = ts.cumsum()
        ts.plot()
        fig_id = 'random_charts'

        df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list("ABCD"))
        df = df.cumsum()
        plt.figure()
        df.plot()
        fig_id1 = 'random_multi_charts'

        algo.save_fig(fig_id)
        # plt.show()
        return render(request, 'introml/ch01/_random_charts.html',
                      {'title': title, 'fig_id': fig_id, 'fig_id1': fig_id1, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png'})


# -------------------------------------------------------------------
def ch02(request):
    title = 'End-to-End Machine Learning Project'
    return render(request, 'introml/ch02.html', {'title': title})


def chapter_2(request, page):
    algo = Algo2()
    if page == "get_data":
        title = "Got the Data"
        housing = algo.DATA
        print('-'*10)
        print('housing.head(20)')
        print(housing.head(20))
        print('-'*10)
        print('housing.info()')
        print(housing.info())
        print('-'*10)
        print('housing["ocean_proximity"].value_counts()')
        print(housing["ocean_proximity"].value_counts())
        print('-'*10)
        print('housing.describe()')
        print(housing.describe())
        print('-'*10)
        return render(request, 'introml/ch02/_get_data.html',
                      {'title': title, 'df': algo.DATA.head(20),
                       'info': housing.info(verbose = False),
                       'frq': pd.DataFrame(housing["ocean_proximity"].value_counts()),
                       'describe': housing.describe()})
    elif page == 'attribute_histogram_plots':
        title = "Histogram of Attributes"
        algo.DATA.hist(bins=50, figsize=(20, 15))
        fig_id = "attribute_histogram_plots"
        algo.save_fig(fig_id)
        return render(request, 'introml/ch02/_attribute_histogram_plots.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID})
    elif page == "california_housing_prices_scatter_plot":
        fig_id = "housing_prices_scatter_plot"
        title = "California housing prices scatter plot"
        algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
        algo.plot(x="longitude", y="latitude", s="population", snf=100,
                  c="median_house_value", fig_name=fig_id, img_name="california", img_type="png")
        return render(request, 'introml/ch02/_california_housing_prices_scatter_plot.html',
                      {'title': title, 'fig_id': "california_"+fig_id, 'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})
    elif page == "data_exploration":

        median_income = algo.DATA['median_income']
        median_income = np.ceil(median_income/1.5)
        print('-'*50)
        print('median_income')
        print(median_income)
        print('-'*50)
        print("median_income.where(median_income > 5, 5.0, inplace = True)")
        median_income.where(median_income < 5, 5.0, inplace = True)
        print('-'*50)
        print('median_income')
        print(median_income)
        print('-'*50)
        print('-'*50)
        print('-'*50)
        type(median_income)
        pd.DataFrame(median_income).hist()

        fig_id_0 = "median_income"
        algo.save_fig(fig_id_0)
        fig_name_0 = "5 Bins for Median Income"

        # ----- Data Exploration -----
        algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
        title = 'Data Exploration'
        train_data = algo.TRAIN.copy()
        corr_matrix = train_data.corr()
        cm = corr_matrix[algo.TARGET_FIELD].sort_values(ascending=False)
        print(cm)
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        scatter_matrix(train_data[attributes], figsize=(12, 8))
        fig_id_1 = "scatter_matrix_plot"
        fig_name_1 = "Scatter matrix plot"
        algo.save_fig(fig_id_1)
        # plt.show()
        # --
        train_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
        plt.axis([0, 16, 0, 550000])
        fig_id_2 = "income_vs_house_value_scatter_plot"
        fig_name_2 = "Income vs house value scatter plot"
        algo.save_fig(fig_id_2)

        train_data["rooms_per_household"] = train_data["total_rooms"] / train_data["households"]
        train_data["bedrooms_per_room"] = train_data["total_bedrooms"] / train_data["total_rooms"]
        train_data["population_per_household"] = train_data["population"] / train_data["households"]

        corr_matrix = train_data.corr()
        corr_target = corr_matrix[algo.TARGET_FIELD].sort_values(ascending=False)
        print(corr_target)

        train_data.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
        plt.axis([0, 5, 0, 520000])
        fig_id_3 = "rooms_per_household_to_median_house_value"
        fig_name_3 = "Rooms per household vs median house value"
        algo.save_fig(fig_id_3)
        # plt.show()
        print(train_data.describe())
        # ----- End Data Exploration -----
        return render(request, 'introml/ch02/_data_exploration.html',
                      {'title': title, 'chapter_id': algo.CHAPTER_ID,
                       'fig_id_0': fig_id_0, 'fig_name_0': fig_name_0,
                       'fig_id_1': fig_id_1, 'fig_name_1': fig_name_1,
                       'fig_id_2': fig_id_2, 'fig_name_2': fig_name_2,
                       'fig_id_3': fig_id_3, 'fig_name_3': fig_name_3,
                       'img_type': 'png'})
    elif page == "assignment":
        title = "Assignment Chapter 2"
        return render(request, 'introml/ch02/_assignment.html',
                      {'title': title, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png'})


# Data preparation for ML
def chapter_2_1(request, page):
    algo = Algo2()
    algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                     bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
    # Prepare the data for Machine Learning algorithms
    print('Prepare the data for Machine Learning algorithms')
    # algo.set_target_data()
    if page == "prepare_the_data_for_ml":
        title = "Prepare the data for ML"
        # Prepare the Data for Machine Learning Algorithms
        print('Prepare the Data for Machine Learning Algorithms')
        # ---- Data Cleaning ---
        print('Data Cleaning')
        print('-'*20)
        housing = algo.TRAIN_DATA.copy()
        sample_incomplete_rows = housing[housing.isnull().any(axis=1)]

        print('-'*10)
        print('-'*10)
        print('-'*10)
        print('-'*10)
        print('-'*10)
        print(sample_incomplete_rows)
        print('-'*10)
        print('-'*10)
        print('-'*10)
        print('-'*10)
        print('-'*10)
        print('-'*10)

        print('-'*10)
        print('sample_incomplete_rows["total_bedrooms"]')
        print('-'*10)
        print(sample_incomplete_rows["total_bedrooms"])
        print('-'*10)
        print('In the code I commented out how to delete rows or columns with missing data')
        # sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
        # sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
        print('-'*10)
        print('Fill missing data with median')
        print('-'*30)
        median = housing["total_bedrooms"].median()

        print('-'*20)
        print('-'*20)
        print('-'*20)
        print(median)
        print('-'*20)
        print('-'*20)
        print('-'*20)

        sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
        print(sample_incomplete_rows["total_bedrooms"])
        print('-'*20)
        # --
        print('Use SimpleImputer')
        print('-'*30)
        housing_num = housing.drop('ocean_proximity', axis=1)

        imputer = SimpleImputer(strategy="median")
        # Remove the text attribute because median can only be calculated on numerical attributes:
        imputer.fit(housing_num)
        print('See that this is the same as manually computing the median of each attribute:')
        print('-'*10)
        print(imputer.statistics_)
        print('-'*10)
        # Check that this is the same as manually computing the median of each attribute:
        print(housing_num.median().values)
        print('-'*20)

        # Transform the training set:
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=algo.TRAIN_DATA.index)
        print('housing_tr.head()')
        print(housing_tr.head())
        print('housing_tr.loc[sample_incomplete_rows.index.values]["total_bedrooms"]')
        print(housing_tr.loc[sample_incomplete_rows.index.values]["total_bedrooms"])
        print('imputer.strategy')
        print(imputer.strategy)
        print('-'*30)

        # -- Handling Text and Categorical Attributes --
        print('Handling Text and Categorical Attributes')
        print("Now let's pre-process the categorical input feature, `ocean_proximity`:")
        print('Use the OneHotEncoder:')
        print('-'*30)
        housing_cat = algo.TRAIN_DATA[['ocean_proximity']]

        cat_encoder = OneHotEncoder(sparse=False)
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

        print('housing_cat_1hot')
        print(housing_cat_1hot)
        # ---- End Data Cleaning ---
        return render(request, 'introml/ch02/_data_processed_successfully.html',
                      {'title': title})
    elif page == "custom_transformers":
        title = "Custom Transformers"
        # --
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False, algo=algo)
        housing_extra_attribs = attr_adder.fit_transform(algo.TRAIN_DATA.values)
        print('housing_extra_attribs')
        print('-'*20)
        print(housing_extra_attribs)
        print('-'*10)
        housing_extra_attribs = pd.DataFrame(
            housing_extra_attribs,
            columns=list(algo.TRAIN_DATA.columns)+["rooms_per_household", "population_per_household"],
            index=algo.TRAIN_DATA.index)
        print('housing_extra_attribs.head()')
        print('-'*20)
        print(housing_extra_attribs.head())
        # --- End Custom Transformers ---
        return render(request, 'introml/ch02/_data_processed_successfully.html',
                      {'title': title})


def chapter_2_2(request, page):
    algo = Algo2()
    algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                     bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
    # algo.set_target_data()
    # --
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder',  CombinedAttributesAdder(algo=algo)),
            ('std_scaler', StandardScaler()),
        ])
    housing_num = algo.TRAIN_DATA.drop('ocean_proximity', axis=1)
    algo.num_attribs = list(housing_num)
    print(algo.num_attribs)
    algo.cat_attribs = ["ocean_proximity"]
    algo.extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    data_pipeline = ColumnTransformer([
            ("num", num_pipeline, algo.num_attribs),
            ("cat", OneHotEncoder(), algo.cat_attribs),
        ])
    algo.set_pipeline(data_pipeline)
    if page == "transformation_pipelines_and_scaling":
        title = "Transformation Pipelines & Scaling"
        housing_num_tr = num_pipeline.fit_transform(housing_num)
        print(title)
        print('housing_num_tr')
        print('-'*10)
        print(housing_num_tr)
        print('data_pipeline')
        print('-'*10)
        print('algo.TRAIN_DATA')
        print(algo.TRAIN_DATA)
        print('-'*10)
        print('algo.TRAIN_DATA.shape')
        print(algo.TRAIN_DATA.shape)
        return render(request, 'introml/ch02/_data_processed_successfully.html', {'title': title})
    elif page == "select_and_train_model":
        title = "Select and Train a Model"
        print(title)
        print('---')
        print('Select and Train a Model')
        print('Training and Evaluating on the Training Set')
        print('lin_reg')
        print('-'*30)
        lin_reg = LinearRegression()
        lin_reg.fit(algo.train_data, algo.TRAIN_TARGET)
        print("let's try the full pre-processing pipeline on a few training instances")
        print('-'*20)
        some_data = algo.TRAIN_DATA.iloc[:5]
        some_labels = algo.TRAIN_TARGET.iloc[:5]
        some_data_prepared = data_pipeline.transform(some_data)
        print("Predictions:", lin_reg.predict(some_data_prepared))
        print('Compare against the actual values:')
        print('-'*10)
        print("Labels:", list(some_labels))
        print(some_data_prepared)
        print('-'*20)

        train_predictions = lin_reg.predict(algo.train_data)
        lin_mse = mean_squared_error(algo.TRAIN_TARGET, train_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print('lin_rmse')
        print(lin_rmse)
        print('-'*20)
        print('Use the object to run models: Not Using Cross-Validation')
        print('-'*30)
        lin_rmse, lin_mae = algo.run_model(model=LinearRegression(), name="linear")
        print("LinearRegression: rmse:  ", lin_rmse, "mae: ", lin_mae)
        print('-'*10)
        tree_reg_rmse, tree_reg_mae = algo.run_model(model=DecisionTreeRegressor(random_state=algo.RANDOM_STATE), name="tree_reg")
        print("DecisionTreeRegressor: rmse: ", tree_reg_rmse, "mae: ", tree_reg_mae)
        print('-'*10)
        forest_reg_rmse, forest_reg_mae = \
            algo.run_model(model=RandomForestRegressor(n_estimators=10, random_state=algo.RANDOM_STATE), name="forest_reg")
        print("RandomForestRegressor: rmse:  ", forest_reg_rmse, "mae: ", forest_reg_mae)
        print('-'*10)
        print('Use the object to run models: Better Evaluation Using Cross-Validation')
        print('-'*30)
        scores = algo.run_model_cv(model=LinearRegression(), cv=10, name="linearCV")
        print("\nLinearRegression:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())

        scores = algo.run_model_cv(model=DecisionTreeRegressor(random_state=algo.RANDOM_STATE), cv=10, name="tree_regCV")
        print("\nDecisionTreeRegressor:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        # --
        scores = algo.run_model_cv(model=RandomForestRegressor(n_estimators=10, random_state=algo.RANDOM_STATE), cv=10,
                                   name="forest_regCV")
        print("\nRandomForestRegressor:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        # -- It takes a long time --
        # scores = algo.run_model_cv(model=SVR(kernel="linear"), cv=10, name="svm_regCV")
        # print("SVR:  \n\n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        return render(request, 'introml/ch02/_data_processed_successfully.html', {'title': title})
    elif page == "fine-tune_your_model_grid_search":
        title = "Fine - Tune Your Model: Grid Search"
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(algo.RANDOM_STATE)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error', return_train_score=True)
        # grid_search.fit(algo.train_data, algo.TRAIN_TARGET)
        df_grid_search, sorted_feature_importances, rmse, confidence_interval = \
            algo.run_model_grid_search_cv(grid_search=grid_search, model_name='GridSearchCV_forest_reg')
        return render(request, 'introml/ch02/_fine-tune_your_model_grid_search.html',
                      {'title': title, 'df_grid_search': df_grid_search})
    elif page == "predictions_with_saved_model":
        title = "Predictions with saved model"
        rmse, confidence_interval = algo.prediction_and_accuracy(model_name='GridSearchCV_forest_reg')
        return render(request, 'introml/ch02/_predictions_with_saved_model.html',
                      {'title': title, 'rmse': rmse, 'confidence_interval': confidence_interval})
    elif page == "fine-tune_your_model_randomized_search":
        title = 'Randomized Grid Search CV'
        print(title)
        print('-'*30)
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=algo.RANDOM_STATE)
        rnd_grid_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                             n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

        df_grid_search, sorted_feature_importances, rmse, confidence_interval = \
            algo.run_model_grid_search_cv(grid_search=rnd_grid_search, model_name='RandomizedSearchCV_forest_reg')
        print(df_grid_search, sorted_feature_importances, rmse, confidence_interval)

        # rnd_search.fit(algo.train_data, algo.TRAIN_TARGET)
        # cvres = rnd_search.cv_results_
        # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        #     print(np.sqrt(-mean_score), params)
        # return render(request, 'introml/ch02/_data_processed_successfully.html',
        #               {'title': title})
        return render(request, 'introml/ch02/_fine-tune_your_model_grid_search.html',
                      {'title': title, 'df_grid_search': df_grid_search})


# -------------------------------------------------------------------
def ch03(request):
    title = 'Ch03: Classification'
    return render(request, 'introml/ch03.html', {'title': title})


def chapter_3(request, page):
    algo = Algo3()
    row_number = 32000
    digit = algo.TRAIN_TARGET[row_number]
    some_digit = algo.TRAIN_DATA[row_number]
    if page == "training_binary_classifier" or page == "confusion_matrix" \
            or page == "comparison_roc" or page == "error_analysis" or page == "multilabel_classification":
        shuffle_index = np.random.permutation(60000)
        X_train, y_train = algo.TRAIN_DATA[shuffle_index], algo.TRAIN_TARGET[shuffle_index]
        y_train_5 = (y_train == 5)
    if page == "get_data":
        title = "Got the Data"
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
        plt.axis("off")
        fig_id = "some_digit_image_" + str(row_number)
        algo.save_fig(fig_id)
        plt.figure(figsize=(9, 9))
        example_images = np.r_[algo.TRAIN_DATA[:12000:600], algo.TRAIN_DATA[13000:30600:600], algo.TRAIN_DATA[30600:60000:590]]
        algo.plot_digits(example_images, images_per_row=10)
        fig_id_10 = "more_digits_plot"
        algo.save_fig(fig_id_10)
        return render(request, 'introml/ch03/_get_data.html',
                      {'title': title,
                       'df': pd.DataFrame(algo.TRAIN_DATA).head(10),
                       'some_digit_image': pd.DataFrame(some_digit_image),
                       'fig_id': fig_id,
                       'chapter_id': algo.CHAPTER_ID,
                       'digit': digit,
                       'img_type': 'png',
                       'fig_id_10': fig_id_10
                       })
    elif page == "training_binary_classifier":
        title = 'Training a Binary Classifier'
        sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
        sgd_clf.fit(X_train, y_train_5)
        print(sgd_clf.predict([some_digit]))
        #
        sgd_clf_cv = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
        #
        never_5_clf = Never5Classifier()
        never_5_clf_cv = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
        #
        return render(request, 'introml/ch03/_training_binary_classifier.html',
                      {'title': title,
                       'sgd_clf_cv': pd.DataFrame(sgd_clf_cv),
                       'never_5_clf_cv': pd.DataFrame(never_5_clf_cv)
                       })
    elif page == "confusion_matrix":
        title = 'Confusion Matrix'
        sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
        confusion_matrix_ = confusion_matrix(y_train_5, y_train_pred)
        print(confusion_matrix_)
        precision_score_ = precision_score(y_train_5, y_train_pred)
        recall_score_ = recall_score(y_train_5, y_train_pred)
        f1_score_ = f1_score(y_train_5, y_train_pred)
        fig_id_1, fig_id_2, fig_id_3, roc_auc_score_ = algo.plot_precision_recall_roc(sgd_clf, X_train, y_train_5, "precision_recall", 3)
        return render(request, 'introml/ch03/_confusion_matrix.html',
                      {'title': title,
                       'confusion_matrix_': pd.DataFrame(confusion_matrix_),
                       'precision_score_': round(10000*precision_score_)/100,
                       'recall_score_': round(10000*recall_score_)/100,
                       'f1_score_': round(10000*f1_score_)/100,
                       'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png',
                       'fig_id_1': fig_id_1,
                       'fig_id_2': fig_id_2,
                       'fig_id_3': fig_id_3,
                       'roc_auc_score_': round(10000*roc_auc_score_)/100
                       })
    elif page == "comparison_roc":
        title = 'Comparison - ROC'
        fig_id = 'comparison_roc_sgd_vs_rf'
        sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
        forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
        algs = {'sgd': {'alg': sgd_clf, 'method': 'decision_function', 'eval': ''},
                'forest': {'alg': forest_clf, 'method': 'predict_proba', 'eval': ''}}
        algo.plot_comparison_roc(algs, X_train, y_train_5, fig_id=fig_id, cv=3)

        return render(request, 'introml/ch03/_comparison_roc.html',
                      {'title': title,
                       'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png',
                       'fig_id': fig_id,
                       })
    elif page == "error_analysis":
        title = 'Error Analysis'
        fig_id = 'error_analysis'
        fig_id_c = fig_id+"_c"
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
        y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
        conf_mx = confusion_matrix(y_train, y_train_pred)
        print('conf_mx')
        print(conf_mx)
        print('-'*50)
        algo.plot_confusion_matrix_gray(conf_mx, fig_id)
        algo.plot_confusion_matrix_color(conf_mx, fig_id_c)

        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        print('norm_conf_mx')
        print(norm_conf_mx)
        print('-'*50)
        fig_id_ep = 'confusion_matrix_errors_plot'
        algo.plot_confusion_matrix_gray(norm_conf_mx, fig_id_ep)

        cl_a, cl_b = 3, 5
        X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
        X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
        X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
        X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

        plt.figure(figsize=(8, 8))
        plt.subplot(221)
        algo.plot_digits(X_aa[:25], images_per_row=5)
        plt.subplot(222)
        algo.plot_digits(X_ab[:25], images_per_row=5)
        plt.subplot(223)
        algo.plot_digits(X_ba[:25], images_per_row=5)
        plt.subplot(224)
        algo.plot_digits(X_bb[:25], images_per_row=5)
        fig_id_ee = "error_analysis_digits_plot"
        algo.save_fig(fig_id_ee)

        return render(request, 'introml/ch03/_error_analysis.html',
                      {'title': title,
                       'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png',
                       'conf_mx': pd.DataFrame(conf_mx),
                       'fig_id': fig_id,
                       'fig_id_c': fig_id_c,
                       'fig_id_ep': fig_id_ep,
                       'fig_id_ee': fig_id_ee,
                       })
    elif page == "multilabel_classification":
        print(1)
        title = 'Multilabel Classification'

        shuffle_index = np.random.permutation(10000)
        X_test, y_test = algo.TEST_DATA[shuffle_index], algo.TEST_TARGET[shuffle_index]

        y_train_large = (y_train >= 7)
        y_train_odd = (y_train % 2 == 1)
        y_multilabel = np.c_[y_train_large, y_train_odd]

        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train, y_multilabel)
        some_digit_ = knn_clf.predict([some_digit])
        print(some_digit_)

        y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
        f1_score(y_multilabel, y_train_knn_pred, average="macro")

        noise = np.random.randint(0, 100, (len(X_train), 784))
        X_train_mod = X_train + noise
        noise = np.random.randint(0, 100, (len(X_test), 784))
        X_test_mod = X_test + noise
        y_train_mod = X_train
        y_test_mod = X_test

        some_index = 5500
        plt.subplot(121)
        algo.plot_digits(X_test_mod[some_index])
        plt.subplot(122);
        algo.plot_digits(y_test_mod[some_index])
        fig_id = "noisy_digit_example_plot"
        algo.save_fig(fig_id)

        knn_clf.fit(X_train_mod, y_train_mod)
        clean_digit = knn_clf.predict([X_test_mod[some_index]])
        algo.plot_digits(clean_digit)
        fig_id_1 = "cleaned_digit_example_plot"
        algo.save_fig(fig_id_1)

        return render(request, 'introml/ch03/_multilabel_classification.html',
                      {'title': title,
                       'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png',
                       'fig_id': fig_id,
                       'fig_id_name': "Noisy digit",
                       'fig_id_1': fig_id_1,
                       'fig_id_1_name': "Cleaned digit",
                       })


# -------------------------------------------------------------------
def ch04(request):
    title = 'Ch04: Model Training'
    return render(request, 'introml/ch04.html', {'title': title})


def chapter_4(request, page):
    algo = Algo4()
    if page == "linear_regression_normal":
        title = "Linear Regression Normal:"
        plt.plot(algo.TRAIN, algo.TRAIN_TARGET, "b.")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.axis([0, 2, 0, 15])
        fig_id1 = "generated_data_plot"
        algo.save_fig(fig_id1)
        # --
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(algo.TRAIN_TARGET)

        y_predict = X_new_b.dot(theta_best)
        print(y_predict)
        # --
        plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
        plt.plot(algo.TRAIN, algo.TRAIN_TARGET, "b.")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.legend(loc="upper left", fontsize=14)
        plt.axis([0, 2, 0, 15])
        fig_id2 = "linear_model_predictions"
        algo.save_fig(fig_id2)
        # --
        lin_reg = LinearRegression()
        lin_reg.fit(algo.TRAIN, algo.TRAIN_TARGET)
        lin_reg.intercept_, lin_reg.coef_
        plr = lin_reg.predict(X_new)
        print(plr)
        return render(request, 'introml/ch04/_linear_regression_normal.html',
                      {'title': title,'chapter_id': algo.CHAPTER_ID, 'img_type': 'png',
                       'fig_id1': fig_id1,
                       'fig_id2': fig_id2,
                       'df_theta_best': pd.DataFrame(theta_best),
                       })
    elif page == "lr_batch_gradient_descent":
        title = "Linear regression using batch gradient descent:"
        eta = 0.1
        n_iterations = 1000
        m = 100
        theta = np.random.randn(2, 1)
        for iteration in range(n_iterations):
            gradients = 2 / m * algo.X_b.T.dot(algo.X_b.dot(theta) - algo.TRAIN_TARGET)
            theta = theta - eta * gradients
        print(theta)
        plr = algo.X_new_b.dot(theta)
        print(plr)
        return render(request, 'introml/_data_processed_successfully.html',
                      {'title': title })
    elif page == "effect_learning_rate":
        title = "Effect Learning Rate:"
        theta_path_bgd = []
        fig_id = algo.show_effect_of_learning_rate()
        return render(request, 'introml/ch04/_effect_learning_rate.html',
                      {'title': title, 'chapter_id': algo.CHAPTER_ID, 'img_type': 'png',
                       'fig_id': fig_id
                       })


# -------------------------------------------------------------------
def proj01(request):
    title = 'Project 1: Dual vs. Single'
    return render(request, 'introml/proj01.html', {'title': title})


def project_1(request, page):
    algo = AlgoP1()
    df = algo.DATA
    df_s = algo.DATA_SOURCE
    df_i = algo.DATA_IMPUTED

    dfs = algo.SINGLE_DATA
    dfd = algo.DUAL_DATA
    dfs_s = algo.SINGLE_DATA_SOURCE
    dfs_i = algo.SINGLE_DATA_IMPUTED
    dfd_s = algo.DUAL_DATA_SOURCE
    dfd_i = algo.DUAL_DATA_IMPUTED
    if page == "data":
        title = "Got the Data for dual & Single"
        describe_df = df.describe()
        describe_ss = dfs_s.describe()
        describe_si = dfs_i.describe()
        describe_ds = dfd_s.describe()
        describe_di = dfd_i.describe()
        return render(request, 'introml/PROJ01/_get_data.html', {'title': title, 'df': df.head(20),
                                                                 'describe_df': describe_df,
                                                                 'describe_ss': describe_ss,
                                                                 'describe_si': describe_si,
                                                                 'describe_ds': describe_ds,
                                                                 'describe_di': describe_di,
                                                                 })
    elif page == "histogram_attributes":
        title = "Histogram of Attributes"
        #
        fig_id_ss = "attribute_histogram_plots_single_source"
        dfs_s.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_ss)
        fig_id_si = "attribute_histogram_plots_single_imputed"
        dfs_i.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_si)
        #
        fig_id_ds = "attribute_histogram_plots_dual_source"
        dfd_s.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_ds)
        fig_id_di = "attribute_histogram_plots_dual_imputed"
        dfd_i.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_di)
        #
        df_miss_per = get_missing_data_percentage(df_s)
        dfs_miss_per = get_missing_data_percentage(dfs_s)
        dfd_miss_per = get_missing_data_percentage(dfd_s)
        #
        row_count_df_s = len(df_s)
        row_count_df_s_na = len(df_s.dropna(how='all'))
        row_count_df_s_per = round(10000*(1-row_count_df_s_na/row_count_df_s))/100

        df_na_miss_per = get_missing_data_percentage(df_s.dropna(how='all'))
        dfs_na_miss_per = get_missing_data_percentage(dfs_s.dropna(how='all'))
        dfd_na_miss_per = get_missing_data_percentage(dfd_s.dropna(how='all'))

        return render(request, 'introml/PROJ01/_attribute_histogram_plots.html',
                      {'title': title,
                       'fig_id_ss': fig_id_ss,
                       'fig_id_si': fig_id_si,
                       'fig_id_ds': fig_id_ds,
                       'fig_id_di': fig_id_di,

                       'chapter_id': algo.CHAPTER_ID,
                       'df_miss_per': df_miss_per,
                       'dfs_miss_per': dfs_miss_per,
                       'dfd_miss_per': dfd_miss_per,

                       'df_na_miss_per': df_na_miss_per,
                       'dfs_na_miss_per': dfs_na_miss_per,
                       'dfd_na_miss_per': dfd_na_miss_per,

                       'row_count_df_s': row_count_df_s,
                       'row_count_df_s_na': row_count_df_s_na,
                       'row_count_df_s_per': row_count_df_s_per

                       })
    elif page == "data_exploration":
        title = "Data Exploration - Scatter Matrix"
        # ----- Data Exploration -----
        fig_id_ss = "scatter_matrix_plot_single_source"
        fig_id_ss_name = "Scatter Matrix Single (Source)"
        scatter_matrix(dfs_s, figsize=(12, 8))
        algo.save_fig(fig_id_ss, tight_layout=False)
        # dfs_i = dfs.iloc[:, 19:]
        # fig_id_si = "scatter_matrix_plot_single_imputed"
        # fig_id_si_name = "Scatter Matrix Single (Imputed)"
        # scatter_matrix(dfs_s, figsize=(12, 8))
        # algo.save_fig(fig_id_si, resolution=750)
        # print(14)
        # ----- End Data Exploration -----
        return render(request, 'introml/PROJ01/_data_exploration.html',
                      {'title': title, 'chapter_id': algo.CHAPTER_ID,
                       'fig_id_ss': fig_id_ss, 'fig_id_ss_name': fig_id_ss_name,
                       # 'fig_id_si': fig_id_si, 'fig_id_si_name': fig_id_si_name,
                       'img_type': 'png'})
    elif page == "corr":
        title = "Correlation analysis"
        corr_matrix_ss = dfs_s.corr()
        corr_matrix_si = dfs_i.corr()
        corr_matrix_ds = dfd_s.corr()
        corr_matrix_di = dfd_i.corr()
        return render(request, 'introml/PROJ01/_corr.html', {'title': title,
                                                                 'corr_matrix_ss': corr_matrix_ss,
                                                                 'corr_matrix_si': corr_matrix_si,
                                                                 'corr_matrix_ds': corr_matrix_ds,
                                                                 'corr_matrix_di': corr_matrix_di,
                                                                 })


def project_1_1(request, page):
    algo = AlgoP1()
    if page == "zig-zag":
        title = "zig_zag"
        # print('algo.SINGLE_DATA_SOURCE.shape', algo.SINGLE_DATA_SOURCE.shape)
        sd_miss_per = get_missing_data_percentage(algo.SINGLE_DATA_SOURCE)
        # print('algo.SINGLE_DATA_SOURCE_ZZ.shape', algo.SINGLE_DATA_SOURCE_ZZ.shape)
        sdz_miss_per = get_missing_data_percentage(algo.SINGLE_DATA_SOURCE_ZZ)
        # print('algo.DUAL_DATA_SOURCE.shape', algo.DUAL_DATA_SOURCE.shape)
        dd_miss_per = get_missing_data_percentage(algo.DUAL_DATA_SOURCE)
        # print('algo.DUAL_DATA_SOURCE_ZZ.shape', algo.DUAL_DATA_SOURCE_ZZ.shape)
        ddz_miss_per = get_missing_data_percentage(algo.DUAL_DATA_SOURCE_ZZ)
        # print('sd', sdz_miss_per, sd_miss_per)
        # print('d', ddz_miss_per, dd_miss_per)

        #
        fig_id_ss = "attribute_histogram_plots_single_source"
        algo.SINGLE_DATA_SOURCE.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_ss)
        fig_id_sszz = "attribute_histogram_plots_single_source_zz"
        algo.SINGLE_DATA_SOURCE_ZZ.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_sszz)

        fig_id_sszzi = "attribute_histogram_plots_single_source_zzi"
        algo.SINGLE_DATA_SOURCE_ZZI.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_sszzi)

        #
        fig_id_ds = "attribute_histogram_plots_dual_source"
        algo.DUAL_DATA_SOURCE.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_ds)
        fig_id_dszz = "attribute_histogram_plots_dual_source_zz"
        algo.DUAL_DATA_SOURCE_ZZ.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_dszz)
        fig_id_dszzi = "attribute_histogram_plots_dual_source_zzi"
        algo.DUAL_DATA_SOURCE_ZZI.hist(bins=50, figsize=(20, 15))
        algo.save_fig(fig_id_dszzi)
        #

        return render(request, 'introml/PROJ01/_zig_zag.html',
                      {'sd_shape': algo.SINGLE_DATA_SOURCE.shape, 'sd_zz_shape': algo.SINGLE_DATA_SOURCE_ZZ.shape,
                       'sd_miss_per' : sd_miss_per, 'sdz_miss_per' : sdz_miss_per,
                       'dd_shape': algo.DUAL_DATA_SOURCE.shape, 'dd_zz_shape': algo.DUAL_DATA_SOURCE_ZZ.shape,
                       'dd_miss_per' : dd_miss_per, 'ddz_miss_per' : ddz_miss_per, 'fig_id_sszzi': fig_id_sszzi,
                       'fig_id_ss': fig_id_ss, 'fig_id_sszz': fig_id_sszz, 'fig_id_ds': fig_id_ds,
                       'fig_id_dszz':fig_id_dszz, 'fig_id_dszzi': fig_id_dszzi,
                       'chapter_id': algo.CHAPTER_ID,
                       })


# -------------------------------------------------------------------
def proj02(request):
    title = 'Project 2: Analysis of securities price'
    return render(request, 'introml/proj02.html', {'title': title})


def get_symbol_data(request):
    symbol_ = request.POST.get('symbol')
    algo = AlgoP2()
    data_ = algo.update_security_prices(symbol=symbol_, start_date="01-01-2000", interval="1d",
                                        test_date_str="01-01-2020", n_window=60, force_run=False,
                                        epochs=25, batch_size=32)
    return render(request, 'introml/PROJ02/_security_data.html',
                  {'symbol': symbol_, 'data': data_, 'fig_id': algo.FIG_ID, 'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})


def project_2(request, page):
    algo = AlgoP2()
    security_type = page.rsplit('_', 1)[1]
    action_type = page.split('_')[0]
    print(security_type)
    if action_type == 'update':
        eval('algo.'+page+'()')
    # if page == "update_symbol_data_funds":
    #     algo.update_symbol_data_funds()
    # if page == "update_symbol_data_sandp":
    #     algo.update_symbol_data_sandp()
    # if page == "update_symbol_data_nasdaq":
    #     algo.update_symbol_data_nasdaq()
    # elif page == "show_symbol_data_funds":
    #     title = "show symbol data: FUNDS"
    # elif page == "show_symbol_data_sandp":
    #     title = "sshow symbol data: S&P"
    # elif page == "show_symbol_data_nasdaq":
    #     title = "show symbol data: NASDAQ"
    securities_ = Security.objects.filter(security_group__group=security_type).all()
    return render(request, 'introml/PROJ02/_update_security_data.html', {'securities': securities_})


# ----------------------------------------------------------------
def gold_1(request, page):
    algo = AlgoG()
    if page == "show excel_data":
        title = "My data"
        df = algo.DATA

        return render(request, 'introml/gold/_get_data.html', {'title': title, 'df': df.head(500), })


def gold(request):
    title = 'Gold App main screen'
    return render(request, 'introml/gold.html', {'title': title})


# ----------------------------------------------------------------
def get_missing_data_percentage(df):
    return round(100*df.isnull().values.ravel().sum()/(df.shape[0]*df.shape[1]), 2)
