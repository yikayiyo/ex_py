#-*- ecoding:utf-8 -*-
'''装饰器'''
import logging
# 简单装饰器
def use_logging(func):
    def wrapper(*args,**kwargs):
        logging.warn("%s is running"%func.__name__)
        return func(*args)
    return wrapper
@use_logging
def foo():
    print "i am foo"

# 有参装饰器
def myhello(tag,*args,**kwargs):
    def real_dec(fn):
        def wrapper(*args,**kwargs):
            if tag=='my':
                print ('this is my special hello')
            else:
                print ('this is a normal hello')
            return fn(*args,**kwargs)
        return wrapper
    return real_dec

@myhello(tag='my')
def hello():
    print 'hello word'



# 简单的类装饰器
class Foo():
    def __init__(self,func):
        # 传入被装饰的函数
        self._func = func
    def __call__(self, *args, **kwargs):
        #调用函数
        logging.warn("%s is running" % self._func.__name__)
        self._func()
@Foo
def cla():
    print('i am cla')


#带参类装饰器
class Goo():
    def __init__(self,tag):
        self._tag = tag
    #如果decorator有参数的话，__init__() 就不能传入fn了，而fn是在__call__的时候传入的
    def __call__(self,fn):
        def wrapper(*args, **kwargs):
            if self._tag=='c':
                print 'complex dec'
            return fn(*args, **kwargs)
        return wrapper
@Goo(tag='c')
def cla(name):
    print('i am %s'%name)


# 经典例子
def memo(fn):
    cache = {}
    miss = object()

    def wrapper(*args):
        result = cache.get(args, miss)
        if result is miss:
            result = fn(*args)
            cache[args] = result
        return result

    return wrapper


@memo
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)



if __name__ == '__main__':
    # foo()
    # bar()
    # cla()
    # hello()
    # cla('LILI')
    print fib(5)