#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng


import os
import json
import codecs
import sys
import operator
import pandas as pd
import numpy as np
import time
from DataQuery import Ticker, Query
import matplotlib as mpl
# import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class Read(object):
    # Read类，读取tradeData各类数据
    def __init__(self):
        """
        初始化
        """
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_folder_path, 'tradeData')
        self.kline_data_path = os.path.join(self.data_path, 'kline')
        self.ticker_data_path = os.path.join(self.data_path, 'ticker')

    def get_freq_types(self):
        """
        /tradeData/kline 中包含的频次
        :return: tradeData中包含的频次
        """
        freq = os.listdir(self.kline_data_path)
        if '.DS_Store' in freq:
            freq.remove('.DS_Store')
            return freq
        else:
            return freq

    def get_market_names(self, freq_type='1d'):
        """
        读取tradeData文件，某个频次中存在的市场
        :param freq_type
        :return: freq_type频次中的市场
        """
        path = os.path.join(self.kline_data_path, freq_type)
        markets = []
        for file in os.listdir(path):
            market_name = file.split('.')[0].replace('-' + freq_type, '')
            if market_name == "":
                continue
            markets.append(market_name)
        return markets

    def read_kline(self, freq_type, market):
        """
        读取单一频次、货币市场的k线数据。
        列标签包括：open, high, low, close, vol, moneyType, symbol
        :param freq_type:数据频次
        :param market:币种市场, 比如btc_qc
        :return: 具有时间索引的金融时间序列数据DataFrame.
        """

        file_dir = os.path.join(self.kline_data_path, freq_type)
        file_name = str.upper(market) + '-' + freq_type + '.json'
        file_path = os.path.join(file_dir, file_name)
        if not os.path.exists(file_path):
            print('文件<%s>不存在，正在获取...' % file_path)
            query = Query()
            query.query_kline(mkt=market, interval=freq_type)
        with codecs.open(os.path.join(file_dir, file_name), 'r', 'utf-8') as f:
            json_data = json.load(f)
        columns = ['open_time', 'open', 'high', 'low', 'close', 'turnover_volume', 'close_time', 'turnover_value',
                   'deal_num', 'bid_volume', 'bid_value', 'other']
        df = pd.DataFrame(json_data['data'], columns=columns)
        columns_to_change_dataType = ['open', 'high', 'low', 'close', 'turnover_volume', 'turnover_value',
                                      'deal_num', 'bid_volume', 'bid_value']
        for c in columns_to_change_dataType:
            df[c] = pd.to_numeric(df[c])
        df['symbol'] = json_data['symbol']
        # df['moneyType'] = json_data['moneyType']
        df['timestamp'] = df['open_time'].apply(lambda x: time.strftime('%Y/%m/%d %X', time.localtime(int(x / 1000))))
        idx = pd.to_datetime(df['timestamp'])
        df = df.set_index(idx)
        df = df.drop(['timestamp'], axis=1)
        return df

    @staticmethod
    def read_ticker_of_quote(quote='USDT'):
        """
        读取最新的ticker数据
        :return:
        """
        ticker = Ticker()
        return ticker.ticker_of_quote(quote=quote)

    def merge_kline(self, freq_type, col='close', market_list=None):
        """
        将相同频次、不同货币市场的某一维度的数据（col）进行合并至同一个数据集
        :param freq_type: 拟合并的数据频次
        :param col: 从 'open','high','low','close','vol' 中进行选择要合并的数据维度
        :param market_list: 可以指定读取哪些货币数据。默认为None，即读取全部
        :return: 按相同时间序列对齐的收盘价数据
        """
        if market_list is None:
            markets = self.get_market_names(freq_type)
        else:
            markets = [str.upper(c) for c in market_list]
        dfs = []
        try:
            for mkt in markets:
                df = self.read_kline(freq_type=freq_type, market=mkt)
                if isinstance(df, pd.DataFrame):
                    symbol = df['symbol'].unique()[0].upper()
                    tmp = df[col].rename(symbol)
                    dfs.append(tmp)
            res = pd.concat(dfs, axis=1)
            return res
        except TypeError:
            print(sys.stderr)
            return None


class Analysis(object):
    """
    市场基本情况分析：
    （1）策略收益与基本收益关系；
    （2）过去24hr涨幅最高货币，以及其过去n天的价格波动；
    （3）过去n天，累计收益表现最好的币，价格波动图；
    （4）货币的相关性分析，构建货币组合，尽量抵消波动风险；
    （5）统计套利
    （6）寻找短线剧烈波动的币
    """

    def __init__(self):
        pass

    @staticmethod
    def top_n_percent_change_markets(num=10, ascending=False):
        """
        24hour内变动百分比最大的n个币
        :param num:
        :param ascending:
        :return:
        """
        ticker = Ticker()
        df = ticker.ticker_of_usdt()
        mkts = ticker.top_n_markets(n=num, col='priceChangePercent', ascending=ascending)
        df_top_n = df[df['symbol'].isin(mkts)][['symbol', 'priceChangePercent']].copy()
        df_top_n = df_top_n.sort_values(by='priceChangePercent', ascending=ascending)
        return df_top_n

    @staticmethod
    def date_filter(df, n=15, start=None, end=None):
        """
        以end_date为结束日期的前n个记录
        :param df: 待筛选的数据集, 索引为时间格式
        :param start: 开始时间
        :param end: 结束时间
        :param n: 前n个日期
        :return: 筛选结果, DataFrame结构
        """
        df = df.copy()
        if start is not None and end is not None:
            df = df.loc[start:end, :].copy()
        elif start is None and end is not None:
            df = df[df.index <= end].copy()
            df = df.tail(n)
        elif start is not None and end is None:
            df = df[df.index >= start].copy()
            df = df.tail(n)
        else:
            df = df.tail(n)
        return df

    @staticmethod
    def price_return_of_last_n(market, interval, start=None, end=None):
        df = Analysis.get_price_return(market=market, interval=interval, start=start, end=end)
        total_return = np.round(np.exp(df['Return'].sum()) - 1, 4)
        return total_return

    def best_performed_of_last_n(self, interval='1d', num=7, start=None, end=None):
        """
        列出过去n个周期收益最高的几个货币及收益率        :param n: 计算收益率的周期数
        :param interval: 时间间隔类别
        :param num: 收益最高的num个货币
        :param start: 开始时间
        :param end: 结束时间
        :return: 收益最高的num个货币及累计收益率
        """
        ticker = Ticker()
        mkts_in_usdt = ticker.top_n_markets(n=100)
        df_price_return = []
        for mkt in mkts_in_usdt:
            total_return = self.price_return_of_last_n(market=mkt, interval=interval, start=start, end=end)
            if total_return == 0:  # 忽略收益为0的货币，因为这通常表示在本时间段内没有数据
                continue
            df_price_return.append((mkt, total_return))
        df_price_return.sort(key=operator.itemgetter(1), reverse=True)

        top_n_mkts = df_price_return[:num]
        df_top_n = pd.DataFrame(top_n_mkts, columns=['symbol', 'return'])
        return df_top_n

    @staticmethod
    def get_price_return(market, interval, start=None, end=None):
        """
        读取k线数据并计算每期价格变动
        :param market:
        :param interval:
        :param start:
        :param end:
        :return:
        """
        read = Read()
        df_kline = read.read_kline(freq_type=interval, market=market)
        df_kline.dropna(subset=['close'], inplace=True)
        df_kline = df_kline.loc[start:end].copy()
        df_kline['Return'] = np.round(np.log(df_kline['close'] / df_kline['close'].shift(1)), 4)
        df_kline.fillna(0, inplace=True)
        df_kline['AccumReturn'] = np.round(df_kline['Return'].cumsum(), 4)
        return df_kline

    def merge_data_for_markets(self, markets, interval, start=None, end=None, col='Return'):
        """
        多个货币每期收益率数据合并
        :param markets:
        :param interval:
        :param start:
        :param end:
        :param col:
        :return:
        """
        dfs = []
        try:
            for mkt in markets:
                df = self.get_price_return(market=mkt, interval=interval, start=start, end=end)
                if len(df) < 1:
                    continue
                df = df[['symbol', col]]
                if isinstance(df, pd.DataFrame):
                    symbol = df['symbol'].unique()[0].upper()[:-4]
                    tmp = df[col].rename(symbol)
                    dfs.append(tmp)
            res = pd.concat(dfs, axis=1)
            res = res.loc[start:end].copy()
            return res
        except TypeError:
            print(sys.stderr)
            return None

    def market_price(self, markets, interval, start=None, end=None):
        df = self.merge_data_for_markets(markets, interval, start=start, end=end, col='close')
        return df

    def individual_return_for_markets(self, markets, interval, start=None, end=None):
        res = self.merge_data_for_markets(markets, interval, start=start, end=end, col='Return')
        res.fillna(0, inplace=True)
        return res

    def accumulative_return_for_markets(self, markets, interval, start=None, end=None):
        res = self.merge_data_for_markets(markets, interval, start=start, end=end, col='AccumReturn')
        res.fillna(method='pad', inplace=True)
        return res

    @staticmethod
    def pyecharts_line_plot(df: pd.DataFrame):
        """
        用pyecharts库绘制累计收益曲线
        :param df:
        :return: None
        """
        x_data = df.index
        line = Line()
        line.set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            datazoom_opts=[opts.DataZoomOpts()],
        )
        line.add_xaxis(xaxis_data=x_data)
        for col in df.columns:
            line.add_yaxis(
                series_name=col,
                y_axis=df[col].values,
                symbol="emptyCircle",
                is_symbol_show=True,
                label_opts=opts.LabelOpts(is_show=False),
            )
        return line

    def lineplot_price(self, markets, interval, start=None, end=None):
        df = self.market_price(markets, interval, start=start, end=end)
        return self.pyecharts_line_plot(df)

    def lineplot_accumulative_returns(self, markets, interval, start=None, end=None):
        df = self.accumulative_return_for_markets(markets, interval, start=start, end=end)
        return self.pyecharts_line_plot(df)

    def lineplot_individual_returns(self, markets, interval, start=None, end=None):
        df = self.individual_return_for_markets(markets, interval, start=start, end=end)
        return self.pyecharts_line_plot(df)

    def plot_top_return_markets(self, interval, num, start=None, end=None):
        """
        画出收益最高的num个货币的收益率曲线
        :param interval:
        :param num:
        :param start
        :param end
        :return:
        """
        df_top_n = self.best_performed_of_last_n(interval=interval, num=num, start=start, end=end)
        markets = list(df_top_n['symbol'].values)
        line = self.lineplot_accumulative_returns(markets=markets, interval=interval, start=start, end=end)
        return line

    def plot_top_pct_change_markets(self, num, interval, start=None, end=None):
        df_top_n = self.top_n_percent_change_markets(num=num, ascending=False)
        markets = list(df_top_n['symbol'].values)
        line = self.accumulative_return_for_markets(markets=markets, interval=interval, start=start, end=end)
        return line

    @staticmethod
    def turnover_value_share(markets=None, start=None, end=None):
        """
        计算一段时间内平均成交额及占比。计算总体为市值在前100的货币
        :param markets: 待列示的货币
        :param start: 开始时间
        :param end: 结束时间
        :return: 各货币的平均成交额及占比
        """
        ticker = Ticker()
        read = Read()
        top_100_mkts = ticker.top_n_markets(n=100)
        merge_turnover = read.merge_kline(col='turnover_value', freq_type='1d', market_list=top_100_mkts)
        merge_turnover = merge_turnover.loc[start:end].copy()
        merge_turnover.fillna(0, inplace=True)
        data = []
        for col in merge_turnover.columns:
            mean_value = merge_turnover[col].mean()
            data.append((col, mean_value))
        df = pd.DataFrame(data, columns=['symbol', 'mean_turnover_value'])
        df.sort_values(by='mean_turnover_value', ascending=False, inplace=True, ignore_index=True)
        df['value_share'] = df['mean_turnover_value'] / df['mean_turnover_value'].sum()
        if markets is not None:
            df = df[df['symbol'].isin(markets)].copy()
        return df
