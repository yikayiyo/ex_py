#-*- encoding:utf-8 -*-
import matplotlib.pyplot as plt
from random_walk import RandomWalk
while True:
    # rw=RandomWalk()
    rw=RandomWalk(5000)
    rw.fill_walk()
    plt.figure(dpi=128,figsize=(10,6))
    points=[range(rw.num_points)]
    plt.scatter(rw.x_values,rw.y_values,c=points,cmap=plt.cm.Blues,s=10)
    # plt.plot(rw.x_values,rw.y_values,linewidth=10)
    #重绘起点终点
    plt.scatter(0,0,c='red',s=100)
    plt.scatter(rw.x_values[-1],rw.y_values[-1],c='green',s=100)
    #隐藏坐标轴
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.show()
    k=raw_input('keep running?(y/n)\n')
    if k=='n':
        break