#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng

from DataRead import Report, Analysis


def test():
    mkt = 'ETHUSDT'
    freq = '1d'
    start_date = '2024-4-1'
    end_date = '2024-5-1'
    report = Report(market=mkt, freq=freq)
    # res = report.price_return_stats(start=start_date, end=end_date)
    # analysis = Analysis()
    # eth_to_btc = analysis.price_to_btc(markets=[mkt], interval=freq, start=start_date, end=end_date, scaled=False)
    report.turnover_value_trend(start_date, end_date)
    return None


if __name__ == '__main__':
    test()
