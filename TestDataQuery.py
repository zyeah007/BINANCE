import os.path
import unittest
import time
import shutil
import json
import pandas as pd
import datetime
from DataQuery import Query
from DataRead import Read


class TestQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.market = 'BTCUSDT'
        self.interval = '1d'

    def test_get_all_ticker(self):
        query = Query()
        query.update_all_ticker()
        file_name = 'all_ticker-{ts}.json'.format(
            ts=time.strftime('%Y%m%d', time.localtime(time.time()))
        )
        data_file_path = os.path.join(os.path.dirname(__file__), 'tradeData/ticker', file_name)
        file_exist = os.path.exists(data_file_path)

        with open(data_file_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data, columns=['symbol', 'price'])
        coins = df['symbol'].values
        self.assertEqual(True, file_exist, msg='数据文件未正确创建!')  # add assertion here
        self.assertIn('ETHBTC', coins, msg='数据内容不正确!')

    def test_query_kline(self):
        query = Query()
        query.query_kline(mkt=self.market, interval=self.interval)
        file_name = '{mkt}-{type}.json'.format(mkt=self.market, type=self.interval)
        file_path = os.path.join(os.path.dirname(__file__), 'tradeData/kline', self.interval, file_name)
        file_exist = os.path.exists(file_path)

        self.assertEqual(True, file_exist, msg='未正确获取到数据!')

    def test_valid_markets(self):
        query = Query()
        valid_markets = query.valid_markets
        self.assertIn('BTCUSDT', valid_markets)

    def test_update_kline(self):
        query = Query()
        query.update_kline_by_markets(markets=[self.market], intervals=[self.interval])

        read = Read()
        kline = read.read_kline(freq_type=self.interval, market=self.market)
        latest_ts = kline.index.max()

        latest_ts = latest_ts.date()
        today_ts = datetime.date.today()
        self.assertEqual(latest_ts, today_ts, msg="K线数据更新后不是最新日期，更新函数有误!")

    def tearDown(self) -> None:
        # 删除测试创建对文件
        ticker_file_dir = os.path.join(os.path.dirname(__file__), 'tradeData/ticker')
        if os.path.exists(ticker_file_dir):
            shutil.rmtree(ticker_file_dir)

        return None


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestQuery('test_valid_markets'))
    suite.addTest(TestQuery('test_update_kline'))
    unittest.TextTestRunner().run(suite)
