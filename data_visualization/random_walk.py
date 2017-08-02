#-*- encoding:utf-8 -*-
from random import choice
class RandomWalk(object):
    '''一个生成随机漫步数据的类'''
    def __init__(self,num_points=5000):
        self.num_points=num_points
        #起始点（0，0）
        self.x_values=[0]
        self.y_values=[0]

    def fill_walk(self):
        '''计算漫步包含的所有点'''
        #直到列表到达指定长度
        while len(self.x_values)<self.num_points:
            #决定方向、步长
            # x_direction=choice([-1,1])
            # x_distance=choice([0,1,2,3,4])
            # x_step=x_direction * x_distance
            # y_direction=choice([-1,1])
            # y_distance=choice([0,1,2,3,4])
            # y_step=y_direction * y_distance
            x_step=self.get_step()
            y_step=self.get_step()
            #不能原地踏步
            if x_step==0 and y_step==0:
                continue
            #计算下一个点的x和y值
            self.x_values.append(self.x_values[-1]+x_step)
            self.y_values.append(self.y_values[-1]+y_step)

    def get_step(self):
        direction=choice([-1,1])
        distance=choice([0,1,2,3,4])
        step=direction*distance
        return step