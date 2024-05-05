import unittest
from YFinanceData import CheckMongoDB, FinanceData


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.symbol = '^DJI'
        self.finance_obj = FinanceData(symbol=self.symbol)

    def test_mongo_start(self):
        mongo_obj = CheckMongoDB()
        mongo_obj.start_mongodb()
        res = mongo_obj.check_mongo_status()
        self.assertEqual(res, True)  # add assertion here
        return None

    def test_add_data_to_mongo(self):
        finance_obj = self.finance_obj
        df_data = finance_obj.get_data(end='2024-3-29')
        finance_obj.add_data(df_data)
        last_timestamp = finance_obj.get_last_date()
        self.assertEqual(last_timestamp, 1711598400)

    def tearDown(self):
        """
        删除测试中创建的整个集合
        :return:
        """
        finance_obj = self.finance_obj
        col = finance_obj.col
        col.drop()
        print('已删除测试中创建的集合<%s>.' % self.symbol)
        return None


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_add_data_to_mongo'))
    unittest.TextTestRunner().run(suite)
