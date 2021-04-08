from __future__ import division
from pandas_datareaders import data
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# get stock ticker symbol from user
stock_symbol = raw_input(
    'Enter ticker symbol for stock: ').upper()

returns = [];
stds = []
for days in [31, 92, 365]:
# set time period for historical prices

    end_time = datetime.today().strftime('%m/%d/%Y')  # current date
    start_time = (datetime.today() -
              timedelta(days=days)
              ).strftime('%m/%d/%Y')

# retreive historical prices for stock
prices = data.DataReader(stock_symbol,
                         data_source='yahoo',
                         start=start_time, end=end_time)

# sort dates in descending order
prices.sort_index(ascending=False, inplace=True)

# calculate daily logarithmic return
prices['Return'] = (np.log(prices['Close'] /
                           prices['Close'].shift(-1)))

# calculate daily standard deviation of returns
d_std = np.std(prices.Return)

# annualize daily standard deviation
std = d_std * 252 ** 0.5

returns.append(list(prices.Return))
stds.append(std)

# Plot histograms
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
n, bins, patches = ax.hist(returns[2][:-1],
                           bins=50, alpha=0.65, color='blue',
                           label='12-month')
n, bins, patches = ax.hist(returns[1][:-1],
                           bins=50, alpha=0.65, color='green',
                           label='3-month')
n, bins, patches = ax.hist(returns[0][:-1],
                           bins=50, alpha=0.65, color='magenta',
                           label='1-month')
ax.set_xlabel('log return of stock price')
ax.set_ylabel('frequency of log return')
ax.set_title('Historical Volatility for ' +
             stock_symbol)

# get x and y coordinate limits
x_corr = ax.get_xlim()
y_corr = ax.get_ylim()

# make room for text
header = y_corr[1] / 5
y_corr = (y_corr[0], y_corr[1] + header)
ax.set_ylim(y_corr[0], y_corr[1])

# print historical volatility on plot
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 30
y = y_corr[1] - (y_corr[1] - y_corr[0]) / 15
ax.text(x, y, 'Annualized Volatility: ',
        fontsize=11, fontweight='bold')
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 15
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '1-month  = ' + str(np.round(stds[0], 3)),
        fontsize=10)
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '3-month  = ' + str(np.round(stds[1], 3)),
        fontsize=10)
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '12-month = ' + str(np.round(stds[2], 3)),
        fontsize=10)

# add legend
ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, -0.1),
          ncol=3, fontsize=11)

# display plot
fig.tight_layout()
fig.show()