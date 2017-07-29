# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import item, Field


class UserItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    follower_count = Field()
    url_token = Field()
    name = Field()