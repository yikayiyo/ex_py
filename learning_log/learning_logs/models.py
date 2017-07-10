# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Topic(models.Model):
    """用户学习的主题,包括主题名和添加日期"""
    text = models.CharField(max_length=200)
    date_added = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.text
class Entry(models.Model):
    """具体知识"""
    topic = models.ForeignKey(Topic)
    text=models.TextField()
    date_added=models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = 'entries'

    def __str__(self):
        """返回模型的字符串表示"""
        if len(self.text)>50:
            return self.text[:50] + "..."
        return self.text
def __unicode__(self):
    """返回模型的字符串表示"""
    return self.text
