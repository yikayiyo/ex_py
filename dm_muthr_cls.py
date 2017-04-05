# -*- ecoding:utf-8 -*-
'''创建Thread的实例，并传递一个可调用的类实例实现多线程'''
import threading
from time import ctime, sleep

loops = [4, 2]


class ThreadFunc(object):
    def __init__(self, func, args, name=''):
        self.name = name
        self.func = func
        self.args = args

    def __call__(self):
        self.func(*self.args)


def loop(nloop, nsec):
    print 'start loop', nloop, 'at', ctime()
    sleep(nsec)
    print 'loop', nloop, 'done at', ctime()


def test():
    print 'starting at:', ctime()
    threads = []
    nloops = range(len(loops))
    # 创建新线程时实例化了ThreadFunc，会调用ThreadFunc的__call__()方法
    for i in nloops:
        t = threading.Thread(target=ThreadFunc(loop, (i, loops[i]), loop.__name__))
        threads.append(t)
    for i in nloops:
        threads[i].start()
    for i in nloops:
        threads[i].join()
    print 'all DONE at', ctime()


if __name__ == '__main__':
    test()
