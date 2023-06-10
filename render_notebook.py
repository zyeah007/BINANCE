#!/usr/bin/env python
# coding=utf-8
# author: charlie_zheng

import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker


def test_render_notebook():
    c = (
        Line()
        .add_xaxis(Faker.choose())
        .add_yaxis("商家A", Faker.values())
        .add_yaxis("商家B", Faker.values())
        .set_global_opts(title_opts=opts.TitleOpts(title="Line-基本示例"))
        .render_notebook()
    )


if __name__ == "main":
    test_render_notebook()
