#-*- ecoding:utf-8 -*-
'''使用锁实现多线程'''
import thread
from time import ctime,sleep

loops=[4,2]
def loop(nloop,nsec,lock):
    print 'start loop',nloop,'at',ctime()
    sleep(nsec)
    print 'loop',nloop,'done at',ctime()
    lock.release()
def main():
    print 'starting at:',ctime()
    locks=[]
    nloops = range(len(loops))
    #上锁
    for i in nloops:
        lock=thread.allocate_lock()
        lock.acquire()
        locks.append(lock)
    #启动线程
    for i in nloops:
        thread.start_new_thread(loop,(i,loops[i],locks[i]))
    #暂停主线程
    for i in nloops:
        while locks[i].locked():pass

    print 'all DONE at:',ctime()
if __name__ == '__main__':
    main()