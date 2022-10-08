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
import numpy as np
import os
import time
import requests
import json
import codecs
import re
from functools import wraps
import operator
from urllib.parse import urlencode
from itertools import product

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


class Ticker(Data):
    def __init__(self):
        super().__init__()
        self.data_type = 'ticker'
        self.url = os.path.join(BASE_URL, 'api/v3/ticker/price')

        self.save_dir = os.path.join(os.path.dirname(__file__), 'tradeData', self.data_type)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_file_name = 'all_ticker-{ts}.json'.format(
            ts=time.strftime('%Y%m%d', time.localtime(self.timestamp))
        )

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

    def top_n_markets(self, n: int):
        """
        获取币值前n位的货币，区分出moneyType 和 symbol
        :param n:
        :return:
        """
        top_n = []
        file = os.path.join(self.save_dir, self.save_file_name)
        if not os.path.exists(file):
            self.save_all_ticker_data()
        with codecs.open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        df = pd.DataFrame(json_data, dtype=np.float64).T
        df['value'] = df['last'] * df['vol']
        if n == 'all':
            top_n_markets = list(df.index)
        else:
            top_n_markets = list(df.sort_values(by='value', ascending=False).head(min(n, len(df))).index)
        for mkt in top_n_markets:
            pattern = r'BTC$|USDT$'
            match = re.search(pattern, mkt)
            # 排除有可能无法匹配的币对
            if match is None:
                continue
            else:
                repl = '_' + match.group()
                mkt = re.sub(pattern, repl, mkt, count=1)
                top_n.append(mkt)
        return top_n


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
        self.base_url = os.path.join(BASE_URL, 'api/v3/klines')

    def get_latest_ts(self):
        """
        获取当前本地数据最新的时间戳
        :return:
        """
        pass

    def get_data(self, startTime=None, endTime=None):
        """

        :param startTime:
        :param endTime:
        :return:
        """
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
        url = os.path.join(BASE_URL, kline_api) + '?' + params
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
                print(count, url)
            elif count >= 10:
                raise TimeoutError()
            else:
                raise TimeoutError()
        r_info.setdefault('symbol', self.market)
        r_info.setdefault('data', json_data)
        return r_info

    def save_data(self):
        """
        更新本地文件保存的交易数据。
        :return:
        """
        base_dir = self.save_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_name = '{mkt}-{type}.json'.format(mkt=self.market, type=self.freq)
        save_path = os.path.join(base_dir, file_name)
        if not os.path.exists(save_path):  # 即原来没有保存过该类型的文件
            json_data = self.get_data()
        else:
            with codecs.open(save_path, 'r', 'utf-8') as old_f:
                old_json_data = json.load(old_f)
                old_data = old_json_data['data']
                # 由于最后一个时间的数据通常不是该时间实际结束时的，故需要剔除
                old_data.sort(key=operator.itemgetter(0))  # 首先将数据根据日期值（即第一个值）排序，然后剔除最后一个
                if len(old_data) > 0:
                    del old_data[-1]
            json_data = self.get_data()
            new_data = json_data['data']
            new_data.sort(key=operator.itemgetter(0))
            new_start_ts = new_data[0][0]  # 新数据的起始时间，列表中第一个数据里的第一个元素
            old_keep_data = [i for i in old_data if i[0] < new_start_ts]
            refreshed_data = old_keep_data + new_data
            del json_data['data']
            json_data['data'] = refreshed_data
        self._save_data(json_data, save_path)
        print('数据已保存:%s' % file_name)
        return None

    def get_historical_data(self, market=None, interval=None, since=None):
        
        pass


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
    def query_kline(mkt, interval):
        """

        :param mkt:
        :param interval:
        :return: json文件
        """
        kline = Kline(market=mkt, freq=interval)
        kline.save_data()

    def update_kline_by_markets(self, markets=None, intervals=None):
        if markets is None:
            markets = self.valid_markets
        if intervals is None:
            intervals = ['1d', '4h', '1w', '3d']
        params_products = product(intervals, markets)
        for param in params_products:
            market = param[1]
            freq = param[0]
            self.query_kline(mkt=market, interval=freq)
            # kline = Kline(market, freq)
            # kline.save_data()
        return None

    def complete_historical_kline_data(self, markets=None, intervals=None):
        """
        补充历史数据至系统最初时间点
        :param markets:
        :param intervals:
        :return:
        """


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


@time_count
def main():
    query = Query()
    query.update_all_ticker()
    query.update_kline_by_markets()


if __name__ == '__main__':
    main()
