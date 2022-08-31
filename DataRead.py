#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng


import os
import json
import codecs
import sys
import pandas as pd
import time


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
        df['timestamp'] = df['open_time'].apply(lambda x: time.strftime('%Y-%m-%d %X', time.localtime(int(x / 1000))))
        idx = pd.to_datetime(df['timestamp'])
        df = df.set_index(idx)
        df = df.drop(['timestamp'], axis=1)
        return df

    def read_ticker(self):
        """
        读取最新的ticker数据
        :return:
        """
        fileList = os.listdir(self.ticker_data_path)
        fileList.sort(reverse=True)
        file = fileList[0]
        file_path = os.path.join(self.ticker_data_path, file)
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data

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
