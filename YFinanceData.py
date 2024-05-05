#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng
# 从yfinance获取金融数据

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/39.0.2171.71 Safari/537.36'  # 后续所有requests请求都将用到都headers
}
SPX = '^SPX'  # 标普500
DJI = '^DJI'  # 道琼斯
IXIC = '^IXIC'  # NASDAQ
NDX = 'ndx'  # 纳斯达克100
GOLD = 'GC=F'  # 黄金价格
SILVER = 'SI=F'  # 白银价格
OIL = 'ROSN.ME'  # 俄罗斯石油
GAS = 'NG=F'  # 天然气

import signal
from functools import wraps
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import subprocess
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line


# import numpy as np


def stop_after(max_seconds):
    """
    如果程序未在规定时间内完成，则停止执行。
    :param max_seconds: 限定程序允许秒数上限
    :return: 装饰器
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            class TimeoutException(Exception):
                pass

            def timeout_handler():
                raise TimeoutException("Execution timed out!")

            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(max_seconds)
                result = func(*args, **kwargs)
                signal.alarm(0)  # 取消定时器
                return result
            except TimeoutException as e:
                print(e)
                raise SystemExit from None  # 如果你想要程序停止，可以使用SystemExit异常

        return wrapper

    return decorator


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


class CheckMongoDB(object):
    """
    MongoDB状态检查及开启
    """

    def __init__(self):
        pass

    @stop_after(5)
    def check_mongodb(self, host='localhost', port=27017):
        try:
            client = MongoClient(host, port)
            client.server_info()
            return True
        except ConnectionFailure:
            return False

    @staticmethod
    def start_mongodb():
        command = ('mongod --dbpath /Users/zhengye/mongodb/data '
                   '--logpath /Users/zhengye/mongodb/log/mongo.log --fork')
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        errors = result.stderr
        # 打印输出和错误信息
        print(output)
        if errors:
            print(errors)
        return None

    def check_mongo_status(self):
        try:
            res = self.check_mongodb()
            if res:
                print("MongoDB is running.")
                return True
        except SystemExit:
            print("Function did not complete within the time limit.")
            print("MongoDB is not running.")
            return False


class FinanceData(object):
    """
    yfinance数据类，包括数据获取，读取及清洗
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.db_name = 'yfinance'
        self.client = MongoClient('mongodb://localhost:27017')
        self.db = self.client[self.db_name]
        self.col = self.db[self.symbol]

    def get_data(self, start='2023-1-1', end='2024-3-29'):
        ticker = yf.Ticker(self.symbol)
        history = ticker.history(start=start, end=end)
        history['Timestamp'] = [int(x.timestamp()) for x in history.index]
        return history

    def save_data(self, df_data):
        """

        :param df_data: 通过yfinance获取的ticker数据，DataFrame形式
        :return: 存储结果
        """
        dict_data = df_data.to_dict(orient='records')
        col = self.col
        client = self.client
        try:
            if col.insert_many(dict_data):
                pass
                # print("保存成功！")
        except ConnectionError:
            print('保存失败。')
        client.close()
        return None

    def add_data(self, df_data):
        """
        1. 先判断数据库是否已存在；
        2. 如果不存在，则直接将数据保存；
        3. 如果已经存在，则获取数据库中最新的一条数据的时间戳：删除数据库中的该条数据；对拟新增数据进行筛选保留，然后存储。
        :param df_data:
        :return:
        """
        db = self.db
        # 1. 如果集合不存在，则直接插入数据
        if self.symbol not in db.list_collection_names():
            data_to_add = df_data
        else:  # 否则，仅插入新的数据
            col = db[self.symbol]
            if col.estimated_document_count() == 0:
                data_to_add = df_data
            else:
                last_timestamp = self.get_last_date()
                data_to_add = df_data[df_data['Timestamp'] >= last_timestamp].copy()
                col.delete_one({'Timestamp': last_timestamp})
        self.save_data(data_to_add)
        return None

    def get_last_date(self):
        """
        获取最后一条数据的时间戳
        :return:
        """
        col = self.col
        last = col.find_one(sort=[('Timestamp', -1)])['Timestamp']
        return last

    def read_data(self):
        col = self.col
        data = col.find()
        df = pd.DataFrame(data=list(data))
        return df

    def daily_yield(self):
        df = self.read_data()
        df['pct_change'] = df['Close'].pct_change()
        df.rename({'pct_change': self.symbol}, inplace=True)
        df.dropna(inplace=True)
        return df[self.symbol]

    def accumlative_yield(self):
        df = self.daily_yield()
        df['accumulative_yield'] = df['pct_change'].cumsum()
        df.rename({'accumulative_yield': self.symbol}, inplace=True)
        return df[self.symbol]

    def plot_daily_yield(self):
        df = self.daily_yield()
        line = pyecharts_line_plot(df=df)
        return line


class Query(object):
    def __init__(self):
        pass

    @staticmethod
    def query_finance_data(start, end, symbols=None, ):
        """

        :param start:
        :param end:
        :param symbols:
        :return:
        """
        for i in tqdm(range(len(symbols)), ncols=90, desc='获取数据...'):
            symbol = symbols[i]
            finance_obj = FinanceData(symbol=symbol)
            df_data = finance_obj.get_data(start=start, end=end)
            finance_obj.add_data(df_data)
        print('获取完成！')
        return None


if __name__ == '__main__':
    data_symbols = [SPX, DJI, IXIC, NDX, GOLD, GAS]
    query = Query()
    mongodb = CheckMongoDB()
    mongodb.start_mongodb()
    start_date = '2020-1-1'
    end_date = '2024-5-2'
    query.query_finance_data(start=start_date, end=end_date, symbols=data_symbols)
