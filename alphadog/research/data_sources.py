# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:33:59 2019

@author: gurpr
"""

from datetime import datetime as dt
import os
import pandas as pd
import pandas_datareader as pdr


start = dt(2019, 1, 1)
end = dt(2019, 6, 1)


##################
# Google finance #
##################
# DEPRECATED
gf = pdr.DataReader('F', 'google', start, end)


##########
# Tiingo #
##########
# Works!

# Get API key
tiingo_api = "04871be6e176dcdaa4222b2cb883ee5aa865e96c"

tgo = pdr.get_data_tiingo('GOOG', start, end, api_key=tiingo_api)

#['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'close',
#       'divCash', 'high', 'low', 'open', 'splitFactor', 'volume']
tgo.columns

tgo.tail()


###############
# Morningstar #
###############
# DEPRECATED
ms = pdr.DataReader('F', 'morningstar', start, end)


#######
# IEX #
#######
iex_api = "sk_f0f11b9b647c430fae60d211bdb30303"
# Works!
iex = pdr.DataReader('F', 'iex', start, end, api_key=iex_api)
iex.columns  # ['open', 'high', 'low', 'close', 'volume']
iex.tail()

'''
There are additional interfaces to this API that are directly exposed: tops (‘iex-tops’) 
and last (‘iex-lasts’). 
A third interface to the deep API is exposed through Deep class or the get_iex_book function.
'''
iex_tops = pdr.DataReader('gs', 'iex-tops') # empty df - why?
iex_tops.tail()


#############
# Robinhood #
#############

rh = pdr.DataReader('F', 'robinhood', start, end)
# Deprecated


##########
# Enigma #
##########
# Mistake in docs - revisit
# Get API key

df = pdr.get_data_enigma('292129b0-1275-44c8-a6a3-2a0881f24fe1', os.getenv('ENIGMA_API_KEY'))


##########
# Quandl #
##########
# Seems to work - current query returns empty df but need to tweak params
# Get API key
quandl_api = "ispyKPWy4qkfbNcxHQKt"
symbol = 'AAPL.US'
ql = pdr.DataReader(symbol, 'quandl', '2019-01-01', '2019-01-05', api_key=quandl_api)


########
# FRED #
########
# Works!

fred = pdr.DataReader('GDP', 'fred', start, end)
fred.columns
fred.tail()


###############
# Fama French #
###############
# Works!
from pandas_datareader.famafrench import get_available_datasets

len(get_available_datasets())

ds = pdr.DataReader('5_Industry_Portfolios', 'famafrench')
ds.keys()
ds[0]


##############
# World Bank #
##############
# Works!

from pandas_datareader import wb
# Compare the Gross Domestic Products per capita in constant dollars in North America
matches = wb.search('gdp.*capita.*const')

# Then use the download function to acquire the data from the World Bank’s servers
dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX'], start=2005, end=2008)
print(dat)
dat['NY.GDP.PCAP.KD'].groupby(level=0).mean()

# Compare GDP to the share of people with cellphone contracts around the world.
wb.search('cell.*%').iloc[:,:2]
ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']
dat = wb.download(indicator=ind, country='all', start=2017, end=2017).dropna()
dat.columns = ['gdp', 'cellphone']
print(dat.tail())


########
# OECD #
########
# Works!

# Trade union density example
tud = pdr.DataReader('TUD', 'oecd', start=start, end=end)
tud.columns
tud.tail()


############
# Eurostat #
############
# Works!

es = pdr.DataReader('tran_sf_railac', 'eurostat')
es.tail()


#######
# TSP #
#######
# Works!

import pandas_datareader.tsp as tsp
tspreader = tsp.TSPReader(start='2018-10-1', end='2018-12-31')
tspdata = tspreader.read()
tspdata.tail()


##################
# NASDAQ Symbols #
##################
# Works!

from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
symbols.columns
symbols.tail()


#########
# Stooq #
#########
# Stale data

stooq = pdr.DataReader('^SPX', 'stooq', start, end)
stooq.columns
stooq.tail()


########
# MOEX #
########
# Works!

moex = pdr.DataReader('USD000UTSTOM', 'moex', start='2018-07-01', end='2018-07-31')
moex.columns
moex.tail()


#########
# Yahoo #
#########
# Deprecated?

from yahoo_finance import Share

yahoo = Share('YHOO')


##########
# Google #
##########
# Deprecated?

from googlefinance import getQuotes

gf = getQuotes('AAPL')


##################
# pandas-finance #
##################
# Works!

import pandas_finance
aapl = pandas_finance.Equity('AAPL')
aapl.adj_close
aapl.close
aapl.annual_dividend
aapl.dividends

aapl.industry
aapl.sector
aapl.trading_data #['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

aapl.profile #risks
aapl.ticker



############
# Intrinio #
############
intrinio_sandbox_api = "OjhjNDA2NDAzZWZkNTA3NzhkMWRkYzNhNDQ0YjRiNzAw"
# TODO