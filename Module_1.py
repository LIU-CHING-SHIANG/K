#!/usr/bin/env python
# coding: utf-8

# In[4]:

# Python Data Analysis for TWSE Stock Market

    
import bs4 as bs
import urllib
import urllib.request
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import mpl_finance as mpf
from sqlalchemy import create_engine

def SQLFetch(start_y, start_m, end_y, end_m, stockNo):
    
    # 轉換日期格式
    def roc_to_ad(date_str):
        parts = date_str.split('/')
        year = int(parts[0]) + 1911
        return str(year) + '-' + parts[1] + '-' + parts[2]    
    
    # 從證交所抓資料
    start_date = datetime(start_y, start_m, 1)
    end_date = datetime(end_y, end_m, 1)

    df_2330 = pd.DataFrame()

    month_list = pd.date_range(start_date, end_date, freq='MS').strftime("%Y%m%d").tolist()

    for month in month_list:
        url_twse = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date='
        url_history = url_twse + month + '&stockNo=' + str(stockNo)
        webpage_history = urllib.request.urlopen(url_history)
        web_html = bs.BeautifulSoup(webpage_history, 'html.parser')

        stock = json.loads(web_html.text)

        stock_info = list(stock.values())
        stock_price = pd.DataFrame(stock_info[4])
        stock_info[3][0] = stock_info[2]
        stock_price.columns = stock_info[3]
        df_price = stock_price.set_index(stock_price.columns[0])
        df_2330 = pd.concat([df_2330, df_price])

    df_2330.columns = ['Volume', 'Turnover', 'Open', 'High','Low', 'Close', 'Spread', 'Transactions']
    df_2330['Volume'] = df_2330['Volume'].str.replace(',', '').astype(int)
    df_2330 = df_2330.apply(pd.to_numeric, errors='coerce')
    df_2330.index = [roc_to_ad(date_str) for date_str in df_2330.index]
    df_2330.index.name = 'date'

    # 存進SQL Database
    engine = create_engine('sqlite:///2330_Data.db', echo = False) 
    df_2330.to_sql('2330_Data', con = engine, if_exists = 'replace', index_label = 'date')

    return df_2330

