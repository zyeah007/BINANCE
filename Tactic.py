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

    @staticmethod
    def buy_or_sell(x):
        if x > 0:
            return "B"
        elif x < 0:
            return "S"
        else:
            return ""

    def deal_direction(self, df: pd.DataFrame):
        self.data_type_check(df=df)
        df['pos_chg'] = df['Position'] - df['Position'].shift(1)
        df['Direction'] = df['pos_chg'].apply(self.buy_or_sell)
        df['Direction'] = df['Direction'].shift(-1)
        return df.copy()

    def base_return(self, df: pd.DataFrame):
        """
        基础收益率，即价格变动累计收益率。
        :param df: pandas.DataFrame数据，index为时间，仅一列.
        :return: 累计基础收益率。
        """
        df = df.copy()
        self.data_type_check(df=df)
        price_return = np.exp(df['Returns'].sum()) - 1
        return price_return

    def tac_return(self, df: pd.DataFrame):
        """
        计算交易策略的累计收益。
        :param df: 包括Position， 每个区间的基础收益率。
        :return: 交易策略的累计收益。
        """
        df = df.copy()
        self.data_type_check(df)
        strategy_return = np.exp(df['Strategy'].sum()) - 1
        return strategy_return

    @staticmethod
    def complete_returns(df: pandas.DataFrame):
        """
        补充几个指标：1）累计价格收益；2）累计策略收益。
        :param df:
        :return:
        """
        if 'Returns' not in df.columns:
            df['Returns'] = np.log(df['close'] / df['close'].shift(1))
            df.fillna(value={'Returns': 0}, inplace=True)
        if 'Strategy' not in df.columns:
            df['Strategy'] = df['Position'] * df['Returns']
        if 'Cum_Strategy_Returns' not in df.columns:
            df['Cum_Strategy_Returns'] = df['Strategy'].cumsum()
        if 'Cum_Price_Returns' not in df.columns:
            df['Cum_Price_Returns'] = df['Returns'].cumsum()
        return df.copy()

    def max_tac_drawdown(self, df: pd.DataFrame):
        """
        计算最大回撤
        :param df:
        :return:
        """
        df = df.copy()
        self.data_type_check(df)
        df = self.complete_returns(df=df)
        max_drawdown = np.exp(df['Cum_Strategy_Returns'].min()) - 1
        return round(max_drawdown, 4)

    def max_price_drop(self, df: pd.DataFrame):
        """
        计算最大的累计价格跌幅
        :param df:
        :return:
        """
        df = df.copy()
        self.data_type_check(df)
        df = self.complete_returns(df=df)
        max_drawdown = np.exp(df['Cum_Price_Returns'].min()) - 1
        return round(max_drawdown, 4)


class SMA(Tactic):
    def __init__(self):
        super().__init__()

    def sma_algorithm(self, df, sma1=12, sma2=26):
        """

        :param df:
        :param sma1:短期
        :param sma2:长期
        :return:
        """
        df = df.copy()
        self.data_type_check(df=df)
        df['sma1'] = df['close'].rolling(window=sma1).mean()
        df['sma2'] = df['close'].rolling(window=sma2).mean()
        df.dropna(inplace=True)
        return df.copy()

    def sma_tac(self, df: pandas.DataFrame, sma1=12, sma2=26, enable_short=True):
        """

        :param df:
        :param sma1:
        :param sma2:
        :param enable_short: 允许做空标识，若为True，则允许空头头寸；若为False，则不允许空头头寸。
        :return:
        """
        df = df.copy()
        short_position = -1
        if not enable_short:
            short_position = 0
        sma = self.sma_algorithm(df, sma1, sma2)
        sma['Position'] = np.where(sma['sma1'] > sma['sma2'], 1, short_position)
        sma['Position'] = sma['Position'].shift(1)
        sma.fillna(value={'Position': 0}, inplace=True)
        sma = self.deal_direction(df=sma)
        sma.dropna(inplace=True)
        return sma.copy()


class MACD(Tactic):
    def __init__(self):
        super().__init__()

    def macd_algorithm(self, df, short=12, long=26, median=9):
        df = df.copy()
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
        df = df.copy()
        short_position = -1
        if not enable_short:
            short_position = 0
        macd = self.macd_algorithm(df, short, long, median)
        macd['Position'] = np.where(macd['dif'] > 0, 1, short_position)
        macd['Position'] = macd['Position'].shift(1)
        macd.fillna(value={'Position': 0}, inplace=True)
        macd = self.deal_direction(df=macd)
        return macd.copy()

    @staticmethod
    def dif_to_dea(x, enable_short=True, to_zero=True):
        """

        :param x:
        :param enable_short:
        :param to_zero: 是否与0线比较。若为True,则还要考虑快慢线是否在零线同侧；否则不用考虑。
        :return:
        """
        short_position = -1
        if not enable_short:
            short_position = 0
        if to_zero:
            if x['dif'] > x['dea'] > 0:
                res = 1
            elif x['dif'] < x['dea'] < 0:
                res = short_position
            else:
                res = 0
        else:
            if x['dif'] >= x['dea']:
                res = 1
            else:
                res = short_position
        return res

    def dea_tac(self, df, short=12, long=26, median=9, enable_short=True, to_zero=True):
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
        :param to_zero:
        :return:
        """
        df = df.copy()
        macd = self.macd_algorithm(df, short, long, median)
        macd['Position'] = macd.apply(lambda x: self.dif_to_dea(x, enable_short=enable_short, to_zero=to_zero), axis=1)
        macd['Position'] = macd['Position'].shift(1)
        macd.fillna(value={'Position': 0}, inplace=True)
        macd = self.deal_direction(df=macd)
        return macd.copy()


class RSI(Tactic):
    pass
