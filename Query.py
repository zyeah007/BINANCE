#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng

# Do data query

from DataQuery import Query, time_count, connection_test


@time_count
def manual_update():
    query = Query()
    # query.update_all_ticker()
    # query.update_kline_by_markets(markets=None, intervals=['3d'], auto=True)
    query.query_top_n_markets(n=100, intervals=['3d'], auto=True)
    # query.complete_historical_kline_data()


@time_count
def auto_update():
    connection_test()
    query = Query()
    query.update_all_ticker()
    # query.update_kline_by_markets(auto=True)
    query.query_top_n_markets(n=100, auto=True)


if __name__ == '__main__':
    manual_update()
    # auto_update()
