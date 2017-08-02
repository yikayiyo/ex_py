#-*- encoding:utf-8 -*-
import matplotlib.pyplot as plt
#数据准备
input_values=[1,2,3,4,5]
squares=[1,4,9,16,25]
plt.plot(input_values,squares,linewidth=5)
#设置图标标题、坐标轴标签
plt.title('Square Numbers',fontsize=25)
plt.xlabel('Index',fontsize=13)
plt.ylabel('Square of Index',fontsize=15)
#设置刻度标记的大小
plt.tick_params(axis='both',labelsize=14)
plt.show()