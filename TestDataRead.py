import unittest
from DataRead import Read


class TestRead(unittest.TestCase):
    def setUp(self) -> None:
        self.market = 'ETHUSDT'
        self.interval = '1d'

    def test_read_kline(self):
        read = Read()
        coin = read.read_kline(freq_type=self.interval, market=self.market)
        symbol = coin['symbol'].unique()[0]
        right_columns = ['open_time', 'open', 'high', 'low', 'close', 'turnover_volume', 'close_time', 'turnover_value',
                         'deal_num', 'bid_volume', 'bid_value', 'other', 'symbol']
        sample_columns = list(coin.columns.values)
        self.assertEqual(symbol, self.market)  # add assertion here
        self.assertListEqual(right_columns, sample_columns)
        self.assertGreater(len(coin), 0)

    def test_merge_kline(self):
        read = Read()
        f_type = '1d'
        col = 'close'
        mkt_list = [
            'BTCUSDT',
            'ETHUSDT',
            'DOTUSDT'
        ]
        merge = read.merge_kline(freq_type=f_type, col=col, market_list=mkt_list)
        columns = list(merge.columns)
        self.assertListEqual(columns, mkt_list)
        self.assertGreater(len(merge), 0)


if __name__ == '__main__':
    unittest.main()
