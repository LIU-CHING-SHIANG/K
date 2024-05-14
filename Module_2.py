#!/usr/bin/env python
# coding: utf-8

# In[6]:

# Python Data Analysis for TWSE Stock Market


import matplotlib
import matplotlib.pyplot as plt
import mpl_finance as mpf
import pandas as pd
from sqlalchemy import create_engine

def FigProcess(df_2330):
    # 連接數據庫
    engine = create_engine('sqlite:///2330_Data.db', echo=False)
    df_2330 = pd.read_sql_query("SELECT * FROM '2330_Data'", engine, index_col = 'date')

    # 計算移動平均線
    sma_20 = df_2330['Close'].rolling(window=20).mean()
    sma_60 = df_2330['Close'].rolling(window=60).mean()

    # 圖大小
    fig = plt.figure(figsize=(24, 15))
    ax = fig.add_axes([0,0.2,1,0.5])
    ax2 = fig.add_axes([0,0,1,0.2])

    # k線圖
    ax.set_xticks(range(0, len(df_2330.index), 20))
    ax.set_xticklabels(df_2330.index[::20])
    mpf.candlestick2_ochl(ax, df_2330['Open'], df_2330['Close'], df_2330['High'],
                              df_2330['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)

    # 畫出移動平均線
    ax.plot(sma_20, label='20-day Moving Average')
    ax.plot(sma_60, label='60-day Moving Average')

    # 成交量
    mpf.volume_overlay(ax2, df_2330['Open'], df_2330['Close'], df_2330['Volume'], colorup='r', colordown='g', width=0.5, alpha=0.8)
    ax2.set_xticks(range(0, len(df_2330.index), 60))
    ax2.set_xticklabels(df_2330.index[::60])
    plt.title("2330 TSMC",fontsize=25, fontweight='bold', loc='center')

    return plt, sma_20, sma_60, ax

