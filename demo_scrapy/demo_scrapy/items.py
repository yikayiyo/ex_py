# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

class DoubanMovieItem(Item):
    # 排名
    ranking = Field()
    # 电影名称
    movie_name = Field()
    # 评分
    score = Field()
    # 评论人数
    score_num = Field()
