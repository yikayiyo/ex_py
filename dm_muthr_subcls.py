# -*- ecoding:utf-8 -*-
'''派生Thread的子类，并创建子类的实例'''
import threading
from time import ctime, sleep

loops = (4, 2)


class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        # 先调用基类的构造函数
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    # 重写run()
    def run(self):
        self.func(*self.args)


def loop(nloop, nsec):
    print 'start loop', nloop, 'at', ctime()
    sleep(nsec)
    print 'loop', nloop, 'done at', ctime()


def test():
    print 'starting at:', ctime()
    threads = []
    nloops = range(len(loops))
    #
    for i in nloops:
        t = MyThread(loop, (i, loops[i]), loop.__name__)
        threads.append(t)
    for i in nloops:
        threads[i].start()
    for i in nloops:
        threads[i].join()
    print 'all DONE at', ctime()


if __name__ == '__main__':
    test()
