#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng
# 对策略结果画图并展示

import matplotlib as mpl
import matplotlib.pyplot as plt
from DataRead import Read
from Tactic import MACD, SMA

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def mark_trade(df, col='close'):
    for idx in df.index:
        mark_text = df.loc[idx, 'Direction']
        x = idx
        y = df.loc[idx, col]
        if mark_text == "B":
            mark_color = 'r'
            plt.scatter(idx, y, c=mark_color, marker='.', s=30)
            plt.annotate(mark_text, xy=(x, y), xytext=(x, y * (1 - 0.005)), weight='bold', color=mark_color)
        else:
            mark_color = 'g'
            plt.scatter(idx, y, c=mark_color, marker='.', s=30)
            plt.annotate(mark_text, xy=(x, y), xytext=(x, y * (1 + 0.005)), weight='bold', color=mark_color)


def plot_dea_return(df):
    trade = df[df['Direction'] != ""].copy()
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    ax1 = plt.subplot(311)
    ax1.plot(df.index, df['Cum_Price_Returns'], color='y', label='Price')
    ax1.plot(df.index, df['Cum_Strategy_Returns'], color='b', label='Strategy')
    mark_trade(trade, 'Cum_Price_Returns')
    ax1.legend(loc='upper left')

    ax4 = ax1.twinx()
    ax4.plot(df.index, df['Position'], c='black', ls=':')
    ax4.set_ylabel('Position', color='black')
    ax4.tick_params('y', colors='r')

    ax2 = plt.subplot(312)
    ax2.plot(df.index, df['close'], c='b')
    mark_trade(trade)
    ax2.set_ylabel('close')

    ax3 = plt.subplot(313)
    plt.axhline(y=0.0, c='black', lw=2, ls='--')
    ax3.plot(df.index, df['dea'], color='g', label='dea')
    ax3.plot(df.index, df['dif'], color='r', label='dif')
    ax3.set_ylabel('macd')
    ax3.legend()

    plt.show()


def plot_sma_return(df):
    trade = df[df['Direction'] != ""].copy()
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    ax1 = plt.subplot(311)
    ax1.plot(df.index, df['Cum_Price_Returns'], color='g', label='Price')
    ax1.plot(df.index, df['Cum_Strategy_Returns'], color='b', label='Strategy')
    mark_trade(trade, 'Cum_Price_Returns')
    ax1.legend(loc='upper left')

    ax4 = ax1.twinx()
    ax4.plot(df.index, df['Position'], c='black', ls=':', label='Position')
    ax4.set_ylabel('Position', color='black')
    ax4.tick_params('y', colors='r')
    ax4.legend(loc='lower left')

    ax2 = plt.subplot(312)
    ax2.plot(df.index, df['close'], c='b')
    mark_trade(df)
    ax2.set_ylabel('close')

    ax3 = plt.subplot(313)
    plt.axhline(y=0.0, c='black', lw=2, ls='--')
    ax3.plot(df.index, df['sma1'], color='g', label='short')
    ax3.plot(df.index, df['sma2'], color='r', label='long')
    ax3.set_ylabel('sma')
    ax3.legend()

    plt.show()
