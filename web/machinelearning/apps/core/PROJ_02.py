import yfinanceng as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from .models import Security, Price, SecurityGroup
from django.db.models import Max
from datetime import timedelta, datetime
import dateutil.parser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import string
from .utlities import Algo

yf.pdr_override()


class AlgoP2(Algo):
    def __init__(self):
        super().__init__(chapter_id="PROJ02_SECURITIES", to_data_path="funds", target_field="price")
        self.SOURCE_WEB_ADDRESS_FUNDS = "https://finance.yahoo.com/screener/predefined/top_mutual_funds?"
        self.SOURCE_WEB_ADDRESS_SANDP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" # "https://www.slickcharts.com/sp500"
        # companies = B
        self.SYMBOL = None
        self.SCALAR = None
        self.GROUP = None
        self.FIG_ID = None

    def split_train_test_by_date(self, date=None, date_column="date", n_window=60):
        self.TRAIN = self.DATA[self.DATA[date_column] < date].copy()
        self.TEST = self.DATA[self.DATA[date_column] >= date].copy()
        train_ = self.TRAIN.drop(['date'], axis=1)
        self.SCALAR = MinMaxScaler()
        train_ = self.SCALAR.fit_transform(train_)
        self.TRAIN_DATA = []
        self.TRAIN_TARGET = []
        for i in range(n_window, train_.shape[0]):
            self.TRAIN_DATA.append(train_[i-n_window:i])
            self.TRAIN_TARGET.append(train_[i, 3])
        self.TRAIN_DATA, self.TRAIN_TARGET = np.array(self.TRAIN_DATA), np.array(self.TRAIN_TARGET)
        past_60_days = self.TRAIN.tail(60)
        self.TEST = past_60_days.append(self.TEST, ignore_index=True)
        test_ = self.TEST.drop(['date'], axis=1)
        # print('test_')
        # print([round(x, 2) for x in self.TEST['adj_close'].values.tolist()])
        test_ = self.SCALAR.transform(test_)
        self.TEST_DATA = []
        self.TEST_TARGET = []
        for i in range(n_window, test_.shape[0]):
            self.TEST_DATA.append(test_[i-n_window:i])
            self.TEST_TARGET.append(test_[i, 3])
        self.TEST_DATA, self.TEST_TARGET = np.array(self.TEST_DATA), np.array(self.TEST_TARGET)
        print(self.TEST_DATA.shape, self.TEST_TARGET.shape)

    def run_rnn_model(self, force_run=True, epochs=20, batch_size=32):
        model = self.create_and_fit_rnn_model(force_run=force_run, epochs=epochs, batch_size=batch_size)
        y_predict = model.predict(self.TEST_DATA)

        y_predict = y_predict*self.SCALAR.data_range_[3] + self.SCALAR.data_min_[3]
        self.TEST_TARGET = self.TEST_TARGET*self.SCALAR.data_range_[3] + self.SCALAR.data_min_[3]
        # print('y_predict')
        # print(y_predict)
        # print('self.TEST_TARGET')
        # print(self.TEST_TARGET)

        plt.figure(figsize=(14, 5))
        plt.plot(y_predict, color='red', label='predict')
        plt.plot(self.TEST_TARGET, color='blue', label='actual')
        plt.title('Security price prediction using RNN for: ' + self.SYMBOL)
        plt.xlabel('time')
        plt.ylabel('security Price')
        plt.legend()
        # plt.show()
        self.save_fig(fig_id=self.FIG_ID, tight_layout=True, fig_extension="png")

    def create_and_fit_rnn_model(self, force_run=True, epochs=20, batch_size=32):
        model_path = os.path.join(self.MODELS_PATH, self.GROUP + '_' + self.SYMBOL + '_rnn.h5')
        print('--1--')
        print(model_path)
        print('--1--')
        if os.path.isfile(model_path) and (force_run is False):
            return tf.keras.models.load_model(model_path)
        model = Sequential()
        model.add(LSTM(units=60, activation='relu', return_sequences=True,
                       input_shape=(self.TRAIN_DATA.shape[1], self.TRAIN_DATA.shape[2])))
        model.add(Dropout(0.2))

        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.TRAIN_DATA, self.TRAIN_TARGET, epochs=epochs, batch_size=batch_size)
        model.save(model_path)
        print('--2--')
        print(model_path)
        print('--2--')
        return model

    # https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460
    def update_symbol_data_funds(self):
        security_group, created = SecurityGroup.objects.get_or_create(group='funds')
        offset = 0
        n = 0
        z = 1
        while z > 0:
            n += 1
            # print('n')
            # print(n)
            path = self.SOURCE_WEB_ADDRESS_FUNDS + "offset=" + str(offset) + "&count=100"
            # print(path)
            # print('n')
            response = requests.get(path)
            soup = BeautifulSoup(response.text, 'html.parser')
            all_a = soup.findAll('a', {"class": "Fw(600)"})
            # print(len(all_a))
            if len(all_a) < 103:
                z = 0
            nn_ = 0
            for a in all_a:
                nn_ += 1
                if nn_ > 3:
                    symbol_ = a['href'].rsplit("/", maxsplit=1)[1]
                    symbol_ = symbol_.rsplit("?", maxsplit=1)[0]
                    title_ = a['title']
                    # print(symbol_, title_)
                    s = Security.objects.create(symbol=symbol_, security_name=title_,
                                                security_group=security_group)
            offset += 100
            time.sleep(1)

    def update_symbol_data_sandp(self):
        security_group, created = SecurityGroup.objects.get_or_create(group='sandp')
        path = self.SOURCE_WEB_ADDRESS_SANDP
        response = requests.get(path)
        soup = BeautifulSoup(response.text, 'html.parser')
        all_tr = soup.findAll('tr')
        z = 0
        for tr in all_tr:
            z += 1
            if z < 2:
                continue
            a_s = tr.select('td > a')
            symbol_ = a_s[0].text
            title_ = a_s[1].text
            p = a_s[1].find_parent('td')
            try:
                # print('p.findChildren()')
                # print(p.findChildren())
                # print('p.contents')
                # print(p.contents)
                # print('p.contents[1]')
                # print(p.contents[1])
                title_ += p.contents[1]
                # print(title_)
            except Exception as ex:
                pass
            s = Security.objects.create(symbol=symbol_, security_name=title_, security_group=security_group)
            if z >505:
                break

    def update_symbol_data_nasdaq(self):
        self.update_symbol_data_for_group(group='nasdaq', url='https://advfn.com/nasdaq/nasdaq.asp?')

    def update_symbol_data_nyse(self):
        self.update_symbol_data_for_group(group='nyse', url='https://www.advfn.com/nyse/newyorkstockexchange.asp?')

    def update_symbol_data_amex(self):
        self.update_symbol_data_for_group(group='amex', url='https://www.advfn.com/amex/americanstockexchange.asp?')

    def update_symbol_data_for_group(self, group='nyse', url='https://www.advfn.com/nyse/newyorkstockexchange.asp?'):
        security_group, created = SecurityGroup.objects.get_or_create(group=group)
        upper_letters = list(string.ascii_uppercase) + ['0']
        # print(upper_letters)
        for l_ in upper_letters:
            path = url + "companies=" + l_
            # print('path')
            # print(path)
            # print('path')
            response = requests.get(path)
            soup = BeautifulSoup(response.text, 'html.parser')
            table_ = soup.find('table', {"class": "market tab1"})
            # print(table_)
            trs = table_.find_all("tr")
            # print(trs)
            z = 0
            for tr in trs:
                z += 1
                if z > 2:
                    try:
                        a_s = tr.select('td > a')
                        symbol_ = a_s[1].text
                        title_ = a_s[0].text
                        # print(symbol_, title_)
                        s = Security.objects.create(symbol=symbol_, security_name=title_,
                                                    security_group=security_group)
                    except Exception as ex:
                        pass
            time.sleep(1)
            print(l_)

    def update_security_prices(self, symbol="FACTX", start_date="01-01-2017", interval="1d",
                               test_date_str="01-01-2020", n_window=60, force_run=True, epochs=20, batch_size=32):
        self.SYMBOL = symbol
        security = Security.objects.filter(symbol=symbol).all()[0]
        self.GROUP = security.security_group.group
        self.FIG_ID = self.GROUP+'_'+self.SYMBOL
        start_date = datetime.strptime(start_date, '%m-%d-%Y').date()
        price_date_max = Price.objects.filter(security=security).aggregate(max_date=Max('price_date'))
        # print('price_date_max')
        # print(price_date_max)
        if price_date_max['max_date']:
            # print("price_date_max['max_date']")
            # print(price_date_max['max_date'])
            start_date = price_date_max['max_date'] + timedelta(days=1)
            # print(type(start_date))
            # print(start_date)
        end_ = datetime.today().date()
        # print('today')
        # print(end_)
        if end_ > start_date:
            # print('update')
            # print(end_)
            # print(start_date)
            data = pdr.get_data_yahoo(symbol, start=start_date, end=end_, interval=interval)
            for i, d in data.iterrows():
                di = str(i.date())
                did = datetime.strptime(di, "%Y-%m-%d").date()
                Price.objects.create(security=security, price_date=did, open=float(d['Open']), high=float(d['High']),
                                     low=float(d['Low']), close=float(d['Close']), adj_close=float(d['AdjClose']),
                                     volume=int(d['Volume']))
        data_ = Price.objects.filter(security=security).all()
        self.DATA = pd.DataFrame.from_records(
            data_.values_list('price_date', 'high', 'low', 'volume', 'adj_close'),
            columns=['date', 'high', 'low', 'volume', 'adj_close']
        )
        test_date = datetime.strptime(test_date_str, '%m-%d-%Y').date()
        self.split_train_test_by_date(date=test_date, date_column="date", n_window=n_window)
        self.run_rnn_model(force_run=force_run, epochs=epochs, batch_size=batch_size)
        return data_
