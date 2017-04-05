# -*- ecoding:utf-8 -*-
'''创建Thread的实例，并传递一个函数实现多线程'''
import threading
from time import ctime, sleep

loops = [4, 2]


def loop(nloop, nsec):
    print 'start loop', nloop, 'at', ctime()
    sleep(nsec)
    print 'loop', nloop, 'done at', ctime()


def test():
    print 'starting at:', ctime()
    threads = []
    nloops = range(len(loops))
    # 实例化Thread时，传入函数和参数
    for i in nloops:
        t = threading.Thread(target=loop, args=(i, loops[i]))
        threads.append(t)
    # 新线程开始执行
    for i in nloops:
        threads[i].start()
    # 等待所有线程运行结束
    for i in nloops:
        threads[i].join()

    print 'all DONE at:', ctime()


if __name__ == '__main__':
    test()
