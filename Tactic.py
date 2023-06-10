#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng
# 几个交易策略及效果回测
import pandas
import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

CURRENT_TACS = {'SMA', 'EMA', 'DEA'}


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
        return df.copy()

    def base_return(self, df: pd.DataFrame):
        """
        基础收益率，即价格变动累计收益率。
        :param df: pandas.DataFrame数据，index为时间，仅一列.
        :return: 累计基础收益率。
        """
        df = df.copy()
        self.data_type_check(df=df)
        if 'Returns' not in df.columns:
            df['Returns'] = np.log(df['close'] / df['close'].shift(1))
            df.fillna(value={'Returns': 0}, inplace=True)
        price_return = np.exp(df['Returns'].sum()) - 1
        return price_return

    def tac_return(self, df: pd.DataFrame, start=None, end=None):
        """
        计算交易策略的累计收益。
        :param df: 包括Position， 每个区间的基础收益率。
        :param start:
        :param end
        :return: 交易策略的累计收益。
        """
        df = df.copy()
        self.data_type_check(df)
        df = self.slice_by_date(df, start_date=start, end_date=end)
        df = self.complete_returns(df)
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

    def max_drawdown(self, df: pd.DataFrame, col='Returns'):
        """

        :param df:
        :param col: 价格Returns，或者策略Strategy
        :return:
        """
        df = df.copy()
        self.data_type_check(df)
        df = self.complete_returns(df=df)
        drawdowns = []
        for i in range(0, len(df) - 2):
            df_test = df.iloc[i:-1].copy()
            df_test['cum'] = df_test[col].cumsum()
            dd = df_test['cum'].min()
            drawdowns.append(dd)
        max_drawdown = round(np.exp(min(drawdowns)) - 1, 4)
        return max_drawdown

    def max_tac_drawdown(self, df: pd.DataFrame):
        """
        计算最大回撤
        :param df:
        :return:
        """
        df = df.copy()
        max_drawdown = self.max_drawdown(df=df, col='Strategy')
        return max_drawdown

    def max_price_drawdown(self, df: pd.DataFrame):
        """
        计算最大的累计价格跌幅
        :param df:
        :return:
        """
        df = df.copy()
        max_drawdown = self.max_drawdown(df=df, col='Returns')
        return max_drawdown

    @staticmethod
    def mark_trade(df, col='close'):
        if len(df) == 0:
            return None
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
        return None

    @staticmethod
    def slice_by_date(df, start_date=None, end_date=None, observe_num=None):
        """
        根据日期对原有数据进行切片
        :param df: 原数据
        :param start_date:开始日期，若为None，则为数据本身最早日期；否则为数据本身日期与start_date孰晚。
        :param end_date:结束日期，若为None，则为数据本身最晚日期；否则为数据本身日期与end」_date孰早。
        :param observe_num: 观察数据量（倒序）。如果为None,则为全部。
        :return: 切片后的数据
        """
        df = df.copy()
        if len(df) == 0:
            return pd.DataFrame()
        first_index = datetime.datetime.strptime(str(df.index[0]), '%Y-%m-%d %X')
        last_index = datetime.datetime.strptime(str(df.index[-1]), '%Y-%m-%d %X')
        if start_date is not None:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            if start < first_index:
                start_date = str(df.index[0])
            else:
                start_date = start_date
        else:
            start_date = str(df.index[0])

        if end_date is not None:
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            if end > last_index:
                end_date = str(df.index[-1])
            else:
                end_date = end_date
        else:
            end_date = str(df.index[-1])
        df = df[start_date:end_date].copy()
        if observe_num is not None:
            if observe_num > len(df):
                df = df.copy()
            else:
                df = df.tail(observe_num).copy()
        return df

    @staticmethod
    def trade_call(df: pd.DataFrame):
        """
        确定交易方向及价格
        :param df: 
        :return: tuple(trade_date, trade_direction, quote_price)
        """
        df = df.copy()
        closing_data = df.iloc[-1].copy()
        trade_date = str(closing_data.name)
        position = closing_data['Position']
        price_return = 100 * (np.exp(closing_data['Returns']) - 1)
        strategy_return = 100 * (np.exp(closing_data['Strategy']) - 1)
        if position == 1:
            deal_direction = 'LONG'
        elif position == -1:
            deal_direction = 'SHORT'
        else:
            deal_direction = 'Close Out'
        quote_price = closing_data['open']
        closing_price = closing_data['close']
        print('Trade Date: ', trade_date)
        print('Deal: ', deal_direction)
        print('Position: ', position)
        if deal_direction != 'Close Out':
            print('Quote Price: ', quote_price)
        print('Closing Price:', closing_price)
        print('Price Return:', '%.2f%%' % price_return)
        print('Strategy Return:', '%.2f%%' % strategy_return)
        return None

    @staticmethod
    def plot_price_return(df: pd.DataFrame):
        df = df.copy()
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax1 = plt.subplot(111)
        ax1.plot(df.index, df['close'], color='r', label='close')
        ax1.set_ylabel('close')
        ax1.legend()
        plt.show()


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
        return sma.copy()

    def plot_sma_return(self, df, start=None, end=None):
        trade = df[df['Direction'] != ""].copy()
        trade = self.slice_by_date(trade, start_date=start, end_date=end)
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))

        ax1 = plt.subplot(311)
        ax1.plot(df.index, df['Cum_Price_Returns'], color='g', label='Price')
        ax1.plot(df.index, df['Cum_Strategy_Returns'], color='b', label='SMA')
        self.mark_trade(trade, 'Cum_Price_Returns')
        ax1.legend(loc='upper left')
        plt.axhline(y=0.0, c='k', lw=2, ls='--')

        ax4 = ax1.twinx()
        ax4.plot(df.index, df['Position'], c='black', ls=':', label='Position')
        ax4.set_ylabel('Position', color='black')
        ax4.tick_params('y', colors='r')
        ax4.legend(loc='lower left')
        plt.axhline(y=0.0, c='r', lw=2, ls='--')

        ax2 = plt.subplot(312)
        ax2.plot(df.index, df['close'], c='b')
        self.mark_trade(df)
        ax2.set_ylabel('close')

        ax3 = plt.subplot(313)
        plt.axhline(y=0.0, c='black', lw=2, ls='--')
        ax3.plot(df.index, df['sma1'], color='g', label='short')
        ax3.plot(df.index, df['sma2'], color='r', label='long')
        ax3.set_ylabel('sma')
        ax3.legend()

        plt.show()


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

    def plot_dea_return(self, df, start=None, end=None):
        """
        绘制DEA策略下的收益图。共绘出3幅图：
        1）该策略收益与基础价格收益的时间曲线比较；
        2）价格走势图并标出该策略下的买卖点位；
        3）买卖信号指标曲线
        :param df: 已计算全部指标后的时间序列数据.
        :param start:
        :param end:
        :return:
        """
        trade = df[df['Direction'] != ""].copy()
        trade = self.slice_by_date(trade, start_date=start, end_date=end)
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))

        ax1 = plt.subplot(311)
        ax1.plot(df.index, df['Cum_Price_Returns'], color='y', label='Price')
        ax1.plot(df.index, df['Cum_Strategy_Returns'], color='b', label='DEA')
        self.mark_trade(trade, 'Cum_Price_Returns')
        ax1.legend(loc='upper left')
        plt.axhline(y=0.0, c='k', lw=2, ls='--')

        ax4 = ax1.twinx()
        ax4.plot(df.index, df['Position'], c='black', ls=':')
        ax4.set_ylabel('Position', color='black')
        ax4.tick_params('y', colors='r')
        plt.axhline(y=0.0, c='r', lw=2, ls='--')

        ax2 = plt.subplot(312)
        ax2.plot(df.index, df['close'], c='b')
        self.mark_trade(trade)
        ax2.set_ylabel('close')

        ax3 = plt.subplot(313)
        # plt.axhline(y=0.0, c='black', lw=2, ls='--')
        ax3.plot(df.index, df['dea'], color='g', label='dea')
        ax3.plot(df.index, df['dif'], color='r', label='dif')
        ax3.set_ylabel('macd')
        ax3.legend()
        plt.show()

        return None

    def plot_ema_return(self, df, start=None, end=None):
        """
        绘制EMA策略下的收益图。共绘出3幅图：
        1）该策略收益与基础价格收益的时间曲线比较；
        2）价格走势图并标出该策略下的买卖点位；
        3）买卖信号指标曲线
        :param df: 已计算全部指标后的时间序列数据.
        :param start:
        :param end
        :return:
        """
        trade = df[df['Direction'] != ""].copy()
        trade = self.slice_by_date(trade, start_date=start, end_date=end)
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))

        ax1 = plt.subplot(311)
        ax1.plot(df.index, df['Cum_Price_Returns'], color='y', label='Price')
        ax1.plot(df.index, df['Cum_Strategy_Returns'], color='b', label='EMA')
        self.mark_trade(trade, 'Cum_Price_Returns')
        ax1.legend(loc='upper left')
        plt.axhline(y=0.0, c='k', lw=2, ls='--')

        ax4 = ax1.twinx()
        ax4.plot(df.index, df['Position'], c='black', ls=':')
        ax4.set_ylabel('Position', color='black')
        ax4.tick_params('y', colors='r')
        plt.axhline(y=0.0, c='r', lw=2, ls='--')

        ax2 = plt.subplot(312)
        ax2.plot(df.index, df['close'], c='b')
        self.mark_trade(trade)
        ax2.set_ylabel('close')

        ax3 = plt.subplot(313)
        plt.axhline(y=0.0, c='black', lw=2, ls='--')
        ax3.plot(df.index, df['dif'], color='r', label='dif')
        ax3.set_ylabel('macd')
        ax3.legend()
        plt.show()

        return None


class RSI(Tactic):
    pass


class MultiTacs(object):
    """
    多个策略的收益比较及收益曲线
    """

    def __init__(self):
        pass

    @staticmethod
    def get_tac_method(tac_name):
        sma = SMA()
        macd = MACD()
        method_dicts = {
            'SMA': sma.sma_tac,
            'DEA': macd.dea_tac,
            'EMA': macd.ema_tac,
        }
        return method_dicts[tac_name]

    @staticmethod
    def _tacs_check(tacs=None):
        if tacs is None:
            raise ValueError('No TAC INPUT. You should at least input one tac name.')
        elif not isinstance(tacs, list):
            raise TypeError('NOT LIST INPUT! You must input tacs as a list!')
        elif not set(tacs) <= CURRENT_TACS:
            raise ValueError("TACS NOT EXIST. You must only input tac names as follows: 'SMA', 'DEA', 'EMA'. ")
        else:
            return None

    def multi_tac_results(self, kline, tacs=None, start=None, end=None, short_flag=True):
        """

        :param kline: 通过导入DataRead.Read类读取Kline数据
        :param tacs:
        :param start:
        :param end:
        :param short_flag:
        :return:
        """
        self._tacs_check(tacs=tacs)
        tacs_results = []
        for tac_name in tacs:
            tactic = Tactic()
            tac = self.get_tac_method(tac_name)
            df_tac = tac(df=kline, enable_short=short_flag)
            df_tac = tactic.slice_by_date(df_tac, start_date=start, end_date=end)
            df_tac = tactic.complete_returns(df_tac)
            tac_res = {
                'tac_name': tac_name,
                'tac_df': df_tac.copy()
            }
            tacs_results.append(tac_res)
        return tacs_results

    def multi_tac_returns(self, kline, tacs=None, start=None, end=None, short_flag=True):
        tacs_results = self.multi_tac_results(kline, tacs=tacs, start=start, end=end, short_flag=short_flag)
        res = []
        for tac in tacs_results:
            tactic = Tactic()
            tac_name = tac['tac_name']
            df_tac = tac['tac_df'].copy()
            tac_return_res = {
                'tac_name': tac_name,
                'price_return': tactic.base_return(df=df_tac),
                'strategy_return': tactic.tac_return(df=df_tac),
                'price_drawdown': tactic.max_price_drawdown(df=df_tac),
                'strategy_drawdown': tactic.max_tac_drawdown(df=df_tac),
            }
            res.append(tac_return_res)
        multi_tac_returns = pd.DataFrame(res)
        multi_tac_returns['strategy_diff'] = multi_tac_returns['strategy_return'] - multi_tac_returns['price_return']
        col = ['tac_name', 'price_return', 'strategy_return', 'strategy_diff', 'price_drawdown', 'strategy_drawdown']
        multi_tac_returns = multi_tac_returns[col].copy()
        multi_tac_returns.sort_values(by=['strategy_return', 'strategy_drawdown'], ascending=[False, False],
                                      inplace=True)
        return multi_tac_returns

    def plot_multi_tac_returns(self, kline, tacs=None, start=None, end=None, short_flag=True):
        tacs_results = self.multi_tac_results(kline, tacs=tacs, start=start, end=end, short_flag=short_flag)
        fig = plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(111)
        df_price = tacs_results[0]['tac_df']
        df_index = df_price.index

        ax1.plot(df_index, df_price['Cum_Price_Returns'], c='k', label='Price')
        for tac in tacs_results:
            tac_name = tac['tac_name']
            df = tac['tac_df']
            ax1.plot(df_index, df['Cum_Strategy_Returns'], label=tac_name)

        plt.axhline(y=0.0, c='black', lw=1, ls='--')
        plt.legend()
        plt.show()
        return None
