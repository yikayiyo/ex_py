#-*- encoding:utf-8 -*-
import matplotlib.pyplot as plt
#point size
# plt.scatter(2,4,s=500)
# x_values=[1,2,3,4,5]
# y_values=[1,4,9,16,25]
x_values=[i for i in range(1001)]
y_values=[x**2 for x in x_values]
plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Blues,s=40)
#设置图标标题、坐标轴标签
plt.title('Square Numbers',fontsize=25)
plt.xlabel('Index',fontsize=13)
plt.ylabel('Square of Index',fontsize=15)
#设置刻度标记的大小
plt.tick_params(axis='both',which='major',labelsize=14)
#坐标轴的取值范围
plt.axis([0,1100,0,1100000])
plt.show()
