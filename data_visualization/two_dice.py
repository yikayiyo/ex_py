#-*- encoding:utf-8 -*-
import pygal
from die import Die
# sides=6
d1=Die()
d2=Die(10)
results=[d1.roll()+d2.roll() for i in range(5000)]
# for roll_num in range(50000):
#     results.append(d1.roll()+d2.roll())
# print results
fres=[results.count(i) for i in range(2,d1.num_sides+d2.num_sides+1)]
# for i in range(2,d1.num_sides+d2.num_sides+1):
#     fres.append(results.count(i))
# print fres
#可视化
hist = pygal.Bar()
hist.title=u"D6和D10同时抛50000次结果分布直方图"
hist.x_labels=[i for i in range(2,d1.num_sides+d2.num_sides+1)]
hist.x_title='Result'
hist.y_title='Frequency of Result'
hist.add('D6+D10',fres)
hist.render_to_file(u'd6+d10.svg')