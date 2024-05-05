#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng

from datetime import datetime
import os
import json
import codecs
import sys
from functools import wraps
import operator
import pandas as pd
import numpy as np
import time
from itertools import *
import math
from statsmodels.tsa.stattools import adfuller as adf_test
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from DataQuery import Ticker, Query
import matplotlib as mpl
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.charts import HeatMap

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

TOP_VALUABLE_MARKETS = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'SOLUSDT',
    'XRPUSDT',
    'DOGEUSDT',
    'ADAUSDT',
    'SHIBUSDT',
    'AVAXUSDT',
    'TRXUSDT',
    'DOTUSDT',
    'BCHUSDT',
    'LINKUSDT',
    'NEARUSDT',
    'MATICUSDT',
    'ICPUSDT',
    'LTCUSDT',
    'DAIUSDT',
    'UNIUSDT',
]

INTERVAL_TO_NUM = {
    '1d': 1,
    '3d': 3,
    '1w': 7,
    '4h': 0.1667,
}


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = codecs.open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 写一个用于记录程序输出日志的装饰器
def process_logger(func):
    __console__ = sys.stdout
    current_folder_path = os.path.dirname(os.path.abspath(__file__))

    @wraps(func)
    def log(*args, **kwargs):
        log_file = os.path.join(current_folder_path, '程序执行日志' + '.txt')  # 核对日志文件存储路径
        sys.stdout = Logger(log_file, sys.stdout)
        print('\n' * 2 + '-' * 30)
        print('\n本次运行时间:%s' % time.strftime('%Y-%m-%d %X', time.localtime(int(time.time()))))
        result = func(*args, **kwargs)
        sys.stdout.log.close()
        sys.stdout = __console__
        print('\n\ndone!')
        return result

    return log


def trend_test(value_series):
    """
    趋势检验
    :param value_series:
    :return: 判断是否存在增大或减少的趋势，或不存在趋势
    """
    adf_res = adf_test(value_series)
    p_value = adf_res[1]
    if p_value < 0.001:
        # p值小，说明序列平稳，即不存在趋势
        test_res = '没有明显变化趋势'
    else:
        # p值较大，说明序列不平稳，存在趋势。
        # 拟合线性回归，通过回归系数符号判断趋势是增加（符号为正）还是减少（符号为负）
        scaled = (value_series - value_series.mean()) / (value_series.max() - value_series.min())
        X_data = [[i + 1] for i in range(len(scaled))]
        model = LinearRegression()
        model.fit(X_data, scaled)
        coef = model.coef_
        if coef > 0:
            test_res = '趋势增加'
        else:
            test_res = '趋势减少'
    return test_res


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

    def performed_rank_of_last_n(self, interval='1d', start=None, end=None):
        """
        计算从 start 到 end 时间段各个货币的收益率，并从高到低排序。
        :param interval:
        :param start:
        :param end:
        :return:
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
        return df_price_return

    def best_performed_of_last_n(self, interval='1d', num=7, start=None, end=None):
        """
        列出过去n个周期收益最高的几个货币及收益率        :param n: 计算收益率的周期数
        :param interval: 时间间隔类别
        :param num: 收益最高的num个货币
        :param start: 开始时间
        :param end: 结束时间
        :return: 收益最高的num个货币及累计收益率
        """
        df = self.performed_rank_of_last_n(interval=interval, start=start, end=end)
        top_n_mkts = df[:num]
        df_top_n = pd.DataFrame(top_n_mkts, columns=['symbol', 'return'])
        return df_top_n

    def worst_performed_of_last_n(self, interval='1d', num=7, start=None, end=None):
        """

        :param interval:
        :param num:
        :param start:
        :param end:
        :return:
        """
        df = self.performed_rank_of_last_n(interval=interval, start=start, end=end)
        df.sort(key=operator.itemgetter(1), reverse=False)
        worst_n_mkts = df[:num]
        df_worst_n = pd.DataFrame(worst_n_mkts, columns=['symbol', 'return'])
        return df_worst_n

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

    @staticmethod
    def price_to_btc(markets, interval='1d', bench_market='BTCUSDT', start=None, end=None, scaled=True):
        """
        价格相对于BTC的比率走势
        :return:
        """
        read = Read()
        if bench_market in markets:
            all_markets = markets
        else:
            all_markets = markets + [bench_market]
        res_df = pd.DataFrame()
        for mkt in all_markets:
            df = read.read_kline(market=mkt, freq_type=interval)[['close']]
            df = df.loc[start:end].copy()
            df.rename(columns={'close': mkt}, inplace=True)
            res_df = pd.concat([res_df, df[[mkt]]], axis=1)
        for mkt in all_markets:
            res_df[mkt] = res_df[mkt] / res_df[bench_market]
            if scaled:
                res_df[mkt] = (res_df[mkt] - res_df[mkt].min()) / (res_df[mkt].max() - res_df[mkt].min())
        # for mkt in markets:
        #     res_df[mkt] = res_df[mkt] / res_df[bench_market]
        #     if scaled:
        #         res_df[mkt] = (res_df[mkt] - res_df[mkt].min()) / (res_df[mkt].max() - res_df[mkt].min())
        # if scaled:
        #     res_df[bench_market] = (res_df[bench_market] - res_df[bench_market].min()) / (
        #             res_df[bench_market].max() - res_df[bench_market].min())
        # else:
        #     res_df.drop(bench_market, axis=1, inplace=True)
        return res_df

    @staticmethod
    def price_volatility(markets, interval='1d', start=None, end=None, window=15):
        """
        计算每期的价格振幅
        :param markets:
        :param interval:
        :param start:
        :param end:
        :param window
        :return: 以时间为索引，每期的价格振幅
        """
        read = Read()
        res_df = pd.DataFrame()
        for mkt in markets:
            kline = read.read_kline(market=mkt, freq_type=interval)
            kline = kline.loc[start:end].copy()
            kline['volatility'] = (kline['high'] - kline['low']) / kline['open']
            kline[mkt] = kline['volatility'].rolling(window=window).mean()
            df = kline[[mkt]]
            res_df = pd.concat([res_df, df], axis=1)
        return res_df

    @staticmethod
    def high_above_close(markets, interval='1d', start=None, end=None, window=15):
        """
        计算每期最高价高出收盘价幅度
        :param markets:
        :param interval:
        :param start:
        :param end:
        :param window:
        :return:
        """
        read = Read()
        res_df = pd.DataFrame()
        for mkt in markets:
            kline = read.read_kline(market=mkt, freq_type=interval)
            kline = kline.loc[start:end].copy()
            kline['high_above_close'] = (kline['high'] - kline['close']) / kline['close']
            kline[mkt] = kline['high_above_close'].rolling(window=window).mean()
            df = kline[[mkt]]
            res_df = pd.concat([res_df, df], axis=1)
        return res_df

    @staticmethod
    def low_below_close(markets, interval='1d', start=None, end=None, window=15):
        """
        计算每期最低价低出收盘价幅度
        :param markets:
        :param interval:
        :param start:
        :param end:
        :param window:
        :return:
        """
        read = Read()
        res_df = pd.DataFrame()
        for mkt in markets:
            kline = read.read_kline(market=mkt, freq_type=interval)
            kline = kline.loc[start:end].copy()
            kline['low_below_close'] = (kline['close'] - kline['low']) / kline['close']
            kline[mkt] = kline['low_below_close'].rolling(window=window).mean()
            df = kline[[mkt]]
            res_df = pd.concat([res_df, df], axis=1)
        return res_df

    def return_diff_pairing(self, interval='1d', start=None, end=None, num=10, least_obs=30, daily_diff_threshold=0.02):
        ticker = Ticker()
        total_markets = ticker.top_n_markets(n=num)
        market_pairs = combinations(total_markets, 2)
        pair_counts = math.factorial(num) // (math.factorial(2) * math.factorial(num - 2))
        res = []
        cols = ['market1', 'market2', 'return_diff', 'daily_diff', 'p_value']
        for pair in tqdm(market_pairs, total=pair_counts, ncols=90, desc='迭代计算中...'):
            test_res = self.return_diff(market_pair=pair, interval=interval, start=start, end=end, least_obs=least_obs)
            mkt_1, mkt_2, diff, daily_diff, diff_std, p_value = test_res
            if diff > 0:
                continue
            else:
                res.append([test_res])
        df = pd.DataFrame(res, columns=cols)
        df = df[df['daily_diff'].abs() >= daily_diff_threshold]
        df = df.sort_values(by=['return_diff'], ascending=False, ignore_index=True)
        return df

    def return_diff(self, market_pair, interval='1d', start=None, end=None, least_obs=30):
        mkt_1 = market_pair[0]
        mkt_2 = market_pair[1]
        merge_return = self.merge_data_for_markets(markets=market_pair, start=start, end=end, interval=interval,
                                                   col='Return')
        merge_return['diff'] = merge_return[mkt_1[:-4]] - merge_return[mkt_2[:-4]]
        # 如果start, end 格式为 datetime.datetime,则转换为strx
        if isinstance(end, datetime):
            end = end.strftime('%Y-%m-%d')
        # ADF检验前，需对待验数据处理：剔除空值，且观测量至少least_obs个
        merge_return.dropna(subset=['diff'], inplace=True)
        if len(merge_return) < least_obs:
            print('观测记录不足<%d>个.' % least_obs)
            return None
        if len(merge_return.loc[end:end]) == 0:  # 结束日期仍有交易数据
            print('所选货币已停止交易!')
            return None
        adf_res = adf_test(merge_return['diff'])
        mean_diff = round(merge_return['diff'].mean(), 4)
        daily_diff = round(merge_return['diff'].resample(rule='1D').sum().mean(), 4)
        diff_std = round(merge_return['diff'].std(), 4)
        p_value = round(adf_res[1], 8)
        if mean_diff < 0:
            _tmp = mkt_1
            mkt_1 = mkt_2
            mkt_2 = _tmp
        res = {
            'market_1': mkt_1,
            'market_2': mkt_2,
            'mean_diff': mean_diff,
            'daily_diff': daily_diff,
            'diff_std': diff_std,
            'p_value': p_value,
        }
        return res

    def price_return_stats(self, market, freq, start, end):
        """
        收益率
        :return:
        """
        df_price_return = self.get_price_return(market=market, interval=freq, start=start, end=end)
        mean_return = df_price_return['Return'].mean()
        accumulative_return = df_price_return['Return'].sum()
        return_std = df_price_return['Return'].std()  # 收益率标准差
        day_diff = INTERVAL_TO_NUM[freq]
        annualized_return_std = return_std * math.sqrt(365/day_diff)
        return mean_return, accumulative_return, return_std, annualized_return_std

    @staticmethod
    def turnover_value_trend(market, freq, start, end):
        """
        判断交易量趋势
        :param market:
        :param freq:
        :param start:
        :param end:
        :return:
        """
        read = Read()
        df = read.read_kline(market=market, freq_type=freq)
        df = df[start:end].copy()
        df.dropna(inplace=True)
        turnovers = df['turnover_value'].values
        # 趋势检验
        if len(turnovers) <= 5:
            test_res = '观测记录过少，无法判断'
            print('<%s>数据量为<%d>,不足以进行<交易量>趋势检验.' % (market + "-" + freq, len(turnovers)))
            return test_res
        test_res = trend_test(turnovers)
        return test_res

    def price_to_btc_trend(self, market, freq, start_date, end_date):
        """
        相对于BTC价格变化趋势
        :param market:
        :param freq:
        :param start_date:
        :param end_date:
        :return:
        """
        price_in_btc = self.price_to_btc(markets=[market], interval=freq, start=start_date, end=end_date,
                                         scaled=False)
        price_in_btc.dropna(inplace=True)
        values = price_in_btc[market].values
        if market == 'BTCUSDT':
            test_res = '无变化'
        else:
            if len(values) <= 5:
                test_res = '观测记录过少，无法判断'
                print('<%s>数据量为<%d>,不足以进行<相对价格>趋势检验.' % (market + "-" + freq, len(values)))
                return test_res
            test_res = trend_test(value_series=values)
        return test_res

    def main_indicators(self, market, freq, start_date, end_date):
        mean_return, accumulative_return, return_std, annualized_return_std = self.price_return_stats(market, freq,
                                                                                                      start_date,
                                                                                                      end_date)
        turnover_test = self.turnover_value_trend(market, freq, start_date, end_date)
        price_trend_test = self.price_to_btc_trend(market, freq, start_date, end_date)
        res = {
            '货币名称': market,
            '平均收益率': round(mean_return, 6),
            '累计收益率': round(accumulative_return, 6),
            '收益波动率': round(return_std, 6),
            '离散系数': round(return_std / abs(mean_return), 6),
            '波动率(年化)': round(annualized_return_std, 6),
            '交易量趋势': turnover_test,
            '相对BTC价格趋势': price_trend_test,
        }
        return res

    def yield_corr_matrix(self, markets, interval='1d', start=None, end=None):
        """
        计算一段时期内markets收益率的两两相关性矩阵
        :param markets:
        :param interval:
        :param start:
        :param end:
        :return:
        """
        df = self.merge_data_for_markets(markets=markets, interval=interval, start=start, end=end)
        correlation_matrix = df.corr(method='pearson')
        correlation_matrix = correlation_matrix.round(4)
        return correlation_matrix

    def yield_corr_to_btc(self, markets, interval='1d', start=None, end=None, window=20):
        df_merge = self.merge_data_for_markets(markets=markets, interval=interval, start=start, end=end)
        data = []
        for mkt in markets:
            cols = df_merge.columns
            mkt = mkt[:-4]
            if mkt not in cols:
                continue
            corr = df_merge[mkt].rolling(window=window).corr(df_merge['BTC'])
            corr.dropna(inplace=True)
            df = pd.DataFrame(corr, columns=[mkt])
            data.append(df)
        res = pd.concat(data, axis=1)
        return res


class Plot(object):
    """
    绘制各类统计图
    """

    def __init__(self):
        pass

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
        analysis = Analysis()
        df = analysis.market_price(markets, interval=interval, start=start, end=end)
        return self.pyecharts_line_plot(df)

    def lineplot_accumulative_returns(self, markets, interval, start=None, end=None):
        analysis = Analysis()
        df = analysis.accumulative_return_for_markets(markets, interval, start=start, end=end)
        return self.pyecharts_line_plot(df)

    def lineplot_individual_returns(self, markets, interval, start=None, end=None):
        analysis = Analysis()
        df = analysis.individual_return_for_markets(markets, interval, start=start, end=end)
        line = self.pyecharts_line_plot(df)
        return line

    def plot_top_return_markets(self, interval, num, start=None, end=None):
        """
        画出收益最高的num个货币的收益率曲线
        """
        analysis = Analysis()
        df_top_n = analysis.best_performed_of_last_n(interval=interval, num=num, start=start, end=end)
        markets = list(df_top_n['symbol'].values)
        line = self.lineplot_accumulative_returns(markets=markets, interval=interval, start=start, end=end)
        return line

    def plot_worst_return_markets(self, interval, num, start=None, end=None):
        analysis = Analysis()
        df_worst_n = analysis.worst_performed_of_last_n(interval=interval, num=num, start=start, end=end)
        markets = list(df_worst_n['symbol'].values)
        line = self.lineplot_accumulative_returns(markets=markets, interval=interval, start=start, end=end)
        return line

    def plot_top_pct_change_markets(self, num, interval, start=None, end=None):
        analysis = Analysis()
        df_top_n = analysis.top_n_percent_change_markets(num=num, ascending=False)
        markets = list(df_top_n['symbol'].values)
        line = self.lineplot_accumulative_returns(markets=markets, interval=interval, start=start, end=end)
        return line

    def plot_price_to_bench(self, markets, interval='1d', bench_market='BTCUSDT', start=None, end=None, scaled=True):
        analysis = Analysis()
        df = analysis.price_to_btc(markets=markets, interval=interval, bench_market=bench_market, start=start, end=end,
                                   scaled=scaled)
        return self.pyecharts_line_plot(df)

    def plot_price_volatility(self, markets, interval='1d', start=None, end=None, window=15):
        analysis = Analysis()
        df = analysis.price_volatility(markets=markets, interval=interval, start=start, end=end, window=window)
        return self.pyecharts_line_plot(df)

    def plot_high_above_close(self, markets, interval='1d', start=None, end=None, window=15):
        analysis = Analysis()
        df = analysis.high_above_close(markets=markets, interval=interval, start=start, end=end, window=window)
        return self.pyecharts_line_plot(df)

    def plot_low_below_close(self, markets, interval='1d', start=None, end=None, window=15):
        analysis = Analysis()
        df = analysis.low_below_close(markets=markets, interval=interval, start=start, end=end, window=window)
        return self.pyecharts_line_plot(df)

    def plot_moving_yield_corr_to_btc(self, markets, interval='1d', start=None, end=None, window=21):
        """
        收益率与BTC的移动相关性
        """
        analysis = Analysis()
        df_corr = analysis.yield_corr_to_btc(markets, interval, start, end, window)
        line = self.pyecharts_line_plot(df_corr)
        return line

    @staticmethod
    def plot_corr_matrix(markets, interval='1d', start=None, end=None):
        """
        Pearson相关性矩阵
        """
        analysis = Analysis()
        matrix = analysis.yield_corr_matrix(markets=markets, interval=interval, start=start, end=end)
        min_corr = int(matrix.min().min() * 10) * 10  # 热力图的下限值
        matrix = matrix * 100
        matrix = matrix.round(2)
        x_value = list(matrix.index.values)
        y_value = list(matrix.columns.values)
        data_value = [[i, j, matrix.iloc[i, j]] for i in range(len(matrix.index)) for j in range(len(matrix.columns))]
        c = (
            HeatMap()
            .add_xaxis(x_value)
            .add_yaxis(
                "相关系数*100", y_value, data_value, label_opts=opts.LabelOpts(position="middle")
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="相关系数矩阵"),
                yaxis_opts=opts.AxisOpts(is_inverse=True),
                visualmap_opts=opts.VisualMapOpts(
                    min_=min_corr, max_=100, is_calculable=True, orient="horizontal", pos_left="center"
                ),
            )
        )
        return c


class Report(object):
    """
    输出指定的分析指标结果，对当前行情进行说明。
    """

    def __init__(self):
        pass

    @process_logger
    def report_markets_indicators(self, markets, interval, start, end):
        data = []
        for mkt in markets:
            analysis = Analysis()
            info = analysis.main_indicators(mkt, freq=interval, start_date=start, end_date=end)
            data.append(info)
        df = pd.DataFrame(data)
        return df


if __name__ == '__main__':
    pass
