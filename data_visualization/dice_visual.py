#-*- encoding:utf-8 -*-
import pygal
from die import Die
# sides=6
d=Die()
results=[]
for roll_num in range(1000):
    res=d.roll()
    results.append(res)
# print results
fres=[]
for i in range(1,d.num_sides+1):
    fres.append(results.count(i))
# print fres
#可视化
hist = pygal.Bar()
hist.title=u"随机抛1000次骰子结果分布直方图"
hist.x_labels=[i for i in range(1,d.num_sides+1)]
hist.x_title='Result'
hist.y_title='Frequency of Result'

hist.add('D6',fres)
hist.render_to_file(u'd6.svg')