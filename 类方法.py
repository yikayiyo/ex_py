class C(object):
'''
类方法可以作为替代的构造函数
多态
结合继承，会有更多的可能性
'''
    def __init__(self,d1):
        self.data = d1

    @classmethod
    def form1(cls,a,b):
        return cls(a+b)

    @classmethod
    def form2(cls,a,b,c):
        return cls(a+b-c)

    # def __str__(self):
    #     return str(self.data)

c1 = C(1)
print(c1)
# 通过实例调用类方法
c2 = c1.form1(2,3)
print(c2)
# 通过类名调用类方法
c3 = C.form2(2,3,4)
print(c3)

'''
<__main__.C object at 0x7f275de93128>
<__main__.C object at 0x7f275de93278>
<__main__.C object at 0x7f275de932b0>
'''
