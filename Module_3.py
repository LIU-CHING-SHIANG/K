#!/usr/bin/env python
# coding: utf-8

# In[4]:

# Python Data Analysis for TWSE Stock Market


def Annotate(plt, df_2330, sma_20, sma_60, ax):
    # 找到全部交叉的日期
    cross_dates = [df_2330.index[i] for i in range(1, len(sma_20)) 
                if (sma_20[i-1] > sma_60[i-1] and sma_20[i] < sma_60[i]) or 
                (sma_20[i-1] < sma_60[i-1] and sma_20[i] > sma_60[i])]

    # 死亡交叉的位置
    death_cross_dates = [df_2330.index[i] for i in range(1, len(sma_20)) 
                        if sma_20[i-1] > sma_60[i-1] and sma_20[i] < sma_60[i]]

    # 標示死亡交叉、黃金交叉
    for date in cross_dates:
        if date in death_cross_dates:
            ax.annotate('Death Cross', xy=(date, sma_20.loc[date]), xytext=(date, sma_20.loc[date]-100),
                        arrowprops=dict(facecolor='b', shrink=0.02))
        else:
            ax.annotate('Golden Cross', xy=(date, sma_20.loc[date]), xytext=(date, sma_20.loc[date]+100),
                        arrowprops=dict(facecolor='b', shrink=0.02))
    ax.legend()
    return plt

