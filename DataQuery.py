#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng

# 本文件包含多种类，用于获取类型交易市场数据
# 获取的市场数据为原始json文件
# 获取数据的base_url: https://api.binance.com
# 获取数据的类别及对应类：
# Ticker: ticker数据
# Kline: k线数据，路径层级为：/kline/数据频率/日期/币种市场,如：/kline/1hour/20201109/btc_qc.json

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/39.0.2171.71 Safari/537.36'  # 后续所有requests请求都将用到都headers
}
import pandas as pd
import os
import time
import sys
import requests
import json
import codecs
from functools import wraps
import operator
from urllib.parse import urlencode, urljoin
from itertools import product
from tqdm import tqdm

BASE_URL = 'https://api.binance.com'


class Data(object):
    # 后面Ticker类、CoinMarket类、Depth类、Kline类的父类
    def __init__(self):
        self.timestamp = int(time.time())

    def _get_data(self, url: str):
        """
        本方法用于从网站获取数据，返回原始json文件。不对json数据进行任何处理。
        :param url: 数据获取的链接
        :return: 原始的json格式数据
        """

        try:
            r = requests.get(url, headers=HEADERS)
            r.raise_for_status()
            r_info = r.json()
            r.close()
            return r_info
        except TimeoutError:
            time.sleep(2)
            self._get_data(url)

    @staticmethod
    def _save_data(data, save_path: str):
        """
        将数据保存至本地
        :param data: 待存储的数据
        :param save_path: 保存至本地文件的路径,json格式文件
        :return: None
        """
        with codecs.open(save_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _save_to_mongo(data):
        """
        数据保存至MongoDB
        :param data:
        :return: None
        """
        pass


class Ticker(Data):
    def __init__(self):
        super().__init__()
        self.data_type = 'ticker'
        self.url = urljoin(BASE_URL, '/api/v3/ticker/24hr')

        self.save_dir = os.path.join(os.path.dirname(__file__), 'tradeData', self.data_type)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_file_name = 'all_24hr_ticker.json'

    def get_all_ticker(self):
        return self._get_data(self.url)

    def save_all_ticker_data(self):
        base_dir = self.save_dir
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        file_name = self.save_file_name
        save_path = os.path.join(base_dir, file_name)
        data = self.get_all_ticker()
        self._save_data(data, save_path)

    def ticker_of_quote(self, quote=None):
        """
        从ticker数据中筛选出报价货币为quote的币
        :param quote:
        :return: 筛选后的DataFrame数据
        """
        base_dir = self.save_dir
        file_name = self.save_file_name
        file_path = os.path.join(base_dir, file_name)
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        df = pd.DataFrame(json_data)
        df['quote_symbol'] = df['symbol'].apply(lambda x: x[-4:])
        df_quote = df[df['quote_symbol'] == quote.upper()].copy()
        # 转换数据格式：因原始数据中没有数值格式，而是np.object格式。
        columns_to_change_data_type = ['priceChange', 'priceChangePercent', 'weightedAvgPrice',
                                       'prevClosePrice', 'lastPrice', 'lastQty', 'bidPrice',
                                       'bidQty', 'askPrice', 'askQty', 'openPrice', 'highPrice', 'lowPrice',
                                       'volume', 'quoteVolume']
        for c in columns_to_change_data_type:
            df_quote[c] = pd.to_numeric(df_quote[c])
        return df_quote

    def ticker_of_usdt(self):
        return self.ticker_of_quote(quote='USDT')

    def top_n_markets(self, n=None, col='quoteVolume', ascending=False):
        """
        获取币值前n位的货币，区分出moneyType 和 symbol
        :param n:
        :param col:
        :param ascending:
        :return:
        """
        df = self.ticker_of_usdt()
        df = df.sort_values(by=[col], ascending=[ascending], ignore_index=True)
        if n is None:
            top_n_markets = list(df['symbol'].values)
        else:
            top_n_markets = list(df.loc[:n - 1, 'symbol'].values)
        return top_n_markets


class Kline(Data):
    # k线数据
    def __init__(self, market, freq):
        """
        具体的币种及k线数据频率
        :param market:币种，如BTCUSDT,必须大写
        :param freq:k线频率，如1m/3m/5m/15m/30m/1d/3d/1w/1h/2h/4h/6h/12hour/1M
        """
        super().__init__()
        self.data_type = 'kline'
        self.market = str.upper(market)
        self.freq = freq
        self.save_dir = os.path.join(os.path.dirname(__file__), 'tradeData', self.data_type,
                                     self.freq)
        self.base_url = urljoin(BASE_URL, 'api/v3/klines')

    def get_latest_ts(self):
        """
        获取当前本地数据最新的时间戳
        :return:
        """
        pass

    def get_url(self, startTime=None, endTime=None):
        kline_api = 'api/v3/klines'
        param_data = {
            'symbol': self.market,
            'interval': self.freq,
            'limit': 1000,
        }
        if startTime is not None:
            startTime = int(time.mktime(time.strptime(startTime, '%Y-%m-%d')) * 1000)
            param_data.setdefault(
                'startTime', startTime
            )
        if endTime is not None:
            endTime = int(time.mktime(time.strptime(endTime, '%Y-%m-%d')) * 1000)
            param_data.setdefault(
                'endTime', endTime
            )
        params = urlencode(param_data)
        url = urljoin(BASE_URL, kline_api) + '?' + params
        return url

    def get_data(self, startTime=None, endTime=None):
        """

        :param startTime:
        :param endTime:
        :return:
        """

        url = self.get_url(startTime, endTime)
        r_info = {}
        json_data = self._get_data(url)
        count = 0
        # 如果连接失败或者返回结果错误，则最多尝试10次
        while True:
            if len(json_data) != 0:
                break
            elif count < 10:
                time.sleep(2)
                json_data = self._get_data(url)
                count += 1
            else:
                raise TimeoutError()
        r_info.setdefault('symbol', self.market)
        r_info.setdefault('data', json_data)
        return r_info

    def manual_get_data(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'tradeData', 'temp',
                                self.freq)
        file_name = '{mkt}-{type}.json'.format(mkt=self.market, type=self.freq)
        file_path = os.path.join(base_dir, file_name)
        r_info = {}
        with codecs.open(file_path, 'r', 'utf-8') as f:
            data = json.load(f)
        r_info['symbol'] = self.market
        r_info['data'] = data
        return r_info

    def save_data(self, startTime=None, endTime=None, auto=True):
        """
        更新本地文件保存的交易数据。
        :param startTime:
        :param endTime:
        :param auto: 自动更新，True；手动更新，False
        :return:
        """
        base_dir = self.save_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_name = '{mkt}-{type}.json'.format(mkt=self.market, type=self.freq)
        save_path = os.path.join(base_dir, file_name)
        if not os.path.exists(save_path):  # 即原来没有保存过该类型的文件
            if auto:
                json_data = self.get_data(startTime=startTime, endTime=endTime)
            else:
                json_data = self.manual_get_data()
        else:
            if auto:
                json_data = self.get_data(startTime=startTime, endTime=endTime)
            else:
                json_data = self.manual_get_data()
            new_data = json_data['data']
            new_data.sort(key=operator.itemgetter(0))
            with codecs.open(save_path, 'r', 'utf-8') as old_f:
                old_json_data = json.load(old_f)
                old_data = old_json_data['data']
                # 由于最后一个时间的数据通常不是该时间实际结束时的，故需要剔除
                old_data.sort(key=operator.itemgetter(0))  # 首先将数据根据日期值（即第一个值）排序，然后剔除最后一个
            # 比较新数据与原数据的时间，确定保留新取得的数据（包括:1)早于原来最早时间的数据和,2)晚于原来最晚时间的数据.)
            old_start_ts = old_data[0][0]

            earlier_kept_data = [i for i in new_data if i[0] < old_start_ts]

            if len(old_data) > 1:
                del old_data[-1]
            old_end_ts = old_data[-1][0]  # 原倒数第二个数据的时间(因原来最后一个数据已被删除)
            later_kept_data = [i for i in new_data if i[0] > old_end_ts]

            refreshed_data = earlier_kept_data + old_data + later_kept_data
            refreshed_data.sort(key=operator.itemgetter(0))
            del json_data['data']
            json_data['data'] = refreshed_data
        self._save_data(json_data, save_path)
        # print('数据已保存:%s' % file_name)
        return None

    def get_historical_data(self):
        base_dir = self.save_dir
        file_name = '{mkt}-{type}.json'.format(mkt=self.market, type=self.freq)
        save_path = os.path.join(base_dir, file_name)
        # 如果从来没有获取过，首先正常获取一遍最新的数据
        if not os.path.exists(save_path):  # 即原来没有保存过该类型的文件
            self.save_data()

        # 获取当前数据的起止日期
        while True:
            with codecs.open(save_path, 'r', 'utf-8') as old_f:
                old_json_data = json.load(old_f)
                old_data = old_json_data['data']
            old_start_ts = old_data[0][0]
            time_delta = 1000 * 24 * 3600 * 1000
            start = old_start_ts - time_delta
            start = time.strftime('%Y-%m-%d', time.localtime(int(start / 1000)))
            early_data = self.get_data(startTime=start)['data']
            early_data.sort(key=operator.itemgetter(0))
            early_start_ts = early_data[0][0]
            if early_start_ts < old_start_ts:
                self.save_data(startTime=start)
            else:
                # print('%s-%s数据已更新至最早时间.' % (self.market, self.freq))
                break


class Query(object):
    # 获取以上各类数据，包括Ticker, Depth, Kline
    def __init__(self):
        pass

    @staticmethod
    def update_all_ticker():
        ticker = Ticker()
        ticker.save_all_ticker_data()

    @property
    def valid_markets(self):
        target_mkt_file = os.path.join(os.path.dirname(__file__), 'tradeData/markets.txt')
        with codecs.open(target_mkt_file, 'r', encoding='utf-8') as f:
            coins = f.readlines()
        mkts = [c.strip() + 'USDT' for c in coins]
        ticker = Ticker()
        ticker_json = ticker.get_all_ticker()
        ticker_df = pd.DataFrame(ticker_json)
        tradePairs = ticker_df['symbol'].values
        valid_markets = set(mkts) & set(tradePairs)
        return valid_markets

    @staticmethod
    def query_kline(mkt, interval, auto=True):
        """

        :param mkt:
        :param interval:
        :param auto:
        :return: json文件
        """
        kline = Kline(market=mkt, freq=interval)
        kline.save_data(auto=auto)

    @staticmethod
    def query_historical_kline(mkt, interval):
        kline = Kline(market=mkt, freq=interval)
        kline.get_historical_data()

    def update_kline_by_markets(self, markets=None, intervals=None, auto=True):
        if markets is None:
            markets = list(self.valid_markets)
        if intervals is None:
            intervals = ['1d', '4h', '1w', '3d']
        # params_products = product(intervals, markets)
        for freq in intervals:
            for i in tqdm(range(len(markets)), ncols=90, desc='获取<%s>频次数据...' % freq):
                market = markets[i]
                self.query_kline(mkt=market, interval=freq, auto=auto)
        # for param in params_products:
        #     market = param[1]
        #     freq = param[0]
        #     self.query_kline(mkt=market, interval=freq, auto=auto)
        return None

    def query_top_n_markets(self, n=50, intervals=None, auto=True):
        ticker = Ticker()
        top_n_markets = ticker.top_n_markets(n=n)
        self.update_kline_by_markets(markets=top_n_markets, intervals=intervals, auto=auto)
        return None

    def complete_historical_kline_data(self, markets=None, intervals=None):
        """
        补充历史数据至系统最初时间点
        :param markets:
        :param intervals:
        :return:
        """
        if markets is None:
            markets = self.valid_markets
        if intervals is None:
            intervals = ['1d', '4h', '1w', '3d']
        params_products = product(intervals, markets)
        for param in params_products:
            market = param[1]
            freq = param[0]
            self.query_historical_kline(mkt=market, interval=freq)
        return None


def time_count(func):
    """
    计算程序运行耗时
    :param func: 实际的程序
    :return: 程序运行用时
    """

    @wraps(func)
    def clock(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        diff = end_time - start_time
        hour_count = int(diff / 3600)
        min_count = int((diff - hour_count * 3600) / 60)
        second_count = int(diff - hour_count * 3600 - min_count * 60)
        raw = [hour_count, min_count, second_count]
        res = []
        for i in raw:
            if i < 10:
                i = '0' + str(i)
            res.append(i)

        print('本次运行共耗时 {H}:{m}:{s}'.format(H=res[0], m=res[1], s=res[2]))

    return clock


def connection_test():
    """
    测试网络连通性。如果正常连接，则继续执行；否则，退出程序。
    :return:
    """
    url = 'https://api.binance.com/api/v3/ping'
    r = requests.get(url)
    data = r.json()
    if data != {}:
        print('无法连接至服务器！')
        sys.exit()
    else:
        print('服务器连接正常。')
    return None
