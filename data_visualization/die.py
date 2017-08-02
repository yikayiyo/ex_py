#-*- encoding:utf-8 -*-
from random import randint
class Die(object):
    def __init__(self,num_sides=6):
        #默认6面
        self.num_sides=num_sides

    def roll(self):
        '''返回一个随机值，1到骰子面数之间'''
        return randint(1,self.num_sides)
