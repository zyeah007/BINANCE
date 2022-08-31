#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng
# 几个交易策略及效果回测

import pandas as pd
import numpy as np


class Tactic(object):
    def __init__(self):
        pass

    @staticmethod
    def data_type_check(df: pd.DataFrame):
        """
        检查数据是否符合：1）数据类型为pandas.DataFrame; 2)数据索引index为pd.core.indexes.datetimes.DatetimeIndex
        :param df: 待校验数据
        :return: 若数据类型有问题，抛出错误；若无，返回None.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError()
        elif not isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise TypeError()
        else:
            return None

    def base_return(self, df: pd.DataFrame):
        """
        基础收益率，即价格变动累计收益率。
        :param df: pandas.DataFrame数据，index为时间，仅一列.
        :return: 累计基础收益率。
        """
        self.data_type_check(df=df)

    def tac_return(self, df: pd.DataFrame):
        """
        计算交易策略的累计收益。
        :param df: 包括Position， 每个区间的基础收益率。
        :return: 交易策略的累计收益。
        """
        self.data_type_check(df)
        df['Strategy_Return'] = df['Position'] * df['Price_Return']
        strategy_return = np.exp(df['Strategy_Return'].sum())
        return strategy_return


class SMA(Tactic):
    def __init__(self):
        super().__init__()

    def sma_algorithm(self, df, sma1, sma2):
        """

        :param df:
        :param sma1:
        :param sma2:
        :return:
        """
        self.data_type_check(df=df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        df['sma1'] = df['close'].rolling(window=sma1).mean()
        df['sma2'] = df['close'].rolling(window=sma2).mean()
        df.dropna(inplace=True)
        df['Position'] = np.where(df['sma1'] > df['sma2'], 1, -1)
        df['Strategy'] = df['Position'] * df['Returns']
        df.dropna(inplace=True)
        return df


class MACD(Tactic):
    def __init__(self):
        super(MACD, self).__init__()

    @staticmethod
    def ema(df: pd.DataFrame, period):
        df['ewm'] = df.ewm(span=period, min_periods=period).mean()
        return df['ewm']

    def macd_algorithm(self, df, short, long, m):
        self.data_type_check(df=df)
        short_ema = self.ema(df, short)
        long_ema = self.ema(df, long)
        dif = short_ema - long_ema
        dea = self.ema(dif, m)
        macd = (dif - dea) * 2
        return macd


class RSI(Tactic):
    pass
