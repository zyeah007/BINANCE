#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng
# 几个交易策略及效果回测
import pandas
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
            raise TypeError('Basic data for calculating should be pandas DataFrame!')
        elif not isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise TypeError('Index for DataFrame should be datetime type.')
        else:
            return None

    def base_return(self, df: pd.DataFrame):
        """
        基础收益率，即价格变动累计收益率。
        :param df: pandas.DataFrame数据，index为时间，仅一列.
        :return: 累计基础收益率。
        """
        self.data_type_check(df=df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        price_return = np.exp(df['Returns'].sum())
        return price_return

    def tac_return(self, df: pd.DataFrame):
        """
        计算交易策略的累计收益。
        :param df: 包括Position， 每个区间的基础收益率。
        :return: 交易策略的累计收益。
        """
        self.data_type_check(df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        df['Strategy'] = df['Position'] * df['Returns']
        strategy_return = np.exp(df['Strategy'].sum())
        return strategy_return

    def max_tac_drawdown(self, df: pd.DataFrame):
        """
        计算最大回撤
        :param df:
        :return:
        """
        self.data_type_check(df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        df['Strategy'] = df['Position'] * df['Returns']
        df['Cum_Stragety_Returns'] = df['Strategy'].cumsum()
        max_drawdown = np.exp(df['Cum_Stragety_Returns'].min()) - 1
        return round(max_drawdown, 4)

    def max_price_drop(self, df: pd.DataFrame):
        """
        计算最大的累计价格跌幅
        :param df:
        :return:
        """
        self.data_type_check(df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        df['Cum_Price_Returns'] = df['Returns'].cumsum()
        max_drawdown = np.exp(df['Cum_Price_Returns'].min()) - 1
        return round(max_drawdown, 4)


class SMA(Tactic):
    def __init__(self):
        super().__init__()

    def sma_algorithm(self, df, sma1, sma2):
        """

        :param df:
        :param sma1:短期
        :param sma2:长期
        :return:
        """
        self.data_type_check(df=df)
        df['Returns'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        df['sma1'] = df['close'].rolling(window=sma1).mean()
        df['sma2'] = df['close'].rolling(window=sma2).mean()
        df.dropna(inplace=True)
        return df.copy()

    def sma_tac(self, df: pandas.DataFrame, sma1, sma2, enable_short=True):
        """

        :param df:
        :param sma1:
        :param sma2:
        :param enable_short: 允许做空标识，若为True，则允许空头头寸；若为False，则不允许空头头寸。
        :return:
        """
        short_position = -1
        if not enable_short:
            short_position = 0
        df = self.sma_algorithm(df, sma1, sma2)
        df['Position'] = np.where(df['sma1'] > df['sma2'], 1, short_position)
        df['Position'] = df['Position'].shift(1)
        df['Strategy'] = df['Position'] * df['Returns']
        df.dropna(inplace=True)
        return df.copy()


class MACD(Tactic):
    def __init__(self):
        super().__init__()

    def macd_algorithm(self, df, short=12, long=26, median=9):
        self.data_type_check(df=df)
        # 计算MACD指标
        df['ema_short'] = df['close'].ewm(span=short, min_periods=short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long, min_periods=long, adjust=False).mean()
        df['dif'] = df['ema_short'] - df['ema_long']
        df['dea'] = df['dif'].ewm(span=median, min_periods=median).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])
        df.dropna(inplace=True)
        return df.copy()

    def ema_tac(self, df, short=12, long=26, median=9, enable_short=True):
        """
        用EMA作为交易指标：
        (1)短期快线由下向上穿越长期慢线，买入信号；
        (2)短期快线由上向下穿越长期慢线，卖出信号。
        :param df:
        :param short:
        :param long:
        :param median:
        :param enable_short
        :return:计算每期头寸（买入或卖出）的数据表
        """
        short_position = -1
        if not enable_short:
            short_position = 0
        macd = self.macd_algorithm(df, short, long, median)
        macd['Position'] = np.where(macd['dif'] > 0, 1, short_position)
        macd['Position'] = macd['Position'].shift(1)
        return macd.copy()

    @staticmethod
    def dif_to_dea(x, enable_short=True):
        short_position = -1
        if not enable_short:
            short_position = 0
        if x['dif']> x['dea'] > 0:
            res = 1
        elif x['dif'] < x['dea'] < 0:
            res = short_position
        else:
            res = 0
        return res

    def dea_tac(self, df, short=12, long=26, median=9, enable_short=True):
        """
        根据DIF快线和DEA慢线的交叉情况判断买卖信号。
        本算法要求比单独DIF（即EMA）策略更严格，要求：
        （1）DIF>0 且 DIF>DEA时（黄金交叉），买入；
        （2）DIF<0 且 DIF<DEA时（死亡交叉），卖出；
        （3）其余波段，不操作。
        :param df:
        :param short:
        :param long:
        :param median:
        :param enable_short
        :return:
        """
        macd = self.macd_algorithm(df, short, long, median)
        macd['Position'] = macd.apply(lambda x: self.dif_to_dea(x, enable_short=enable_short), axis=1)
        macd['Position'] = macd['Position'].shift(1)
        return macd.copy()


class RSI(Tactic):
    pass
