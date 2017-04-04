#-*- ecoding:utf-8 -*-
'''利用thread的多线程机制，两个循环并发执行，总时间与最慢的那个线程相关'''
import thread
from time import ctime,sleep

def loop0():
    print 'start loop0 at:',ctime()
    sleep(4)
    print 'loop 0 done at:',ctime()

def loop1():
    print 'start loop1 at:', ctime()
    sleep(2)
    print 'loop 1 done at:', ctime()

def main():
    print 'starting at:',ctime()
    thread.start_new_thread(loop0,())
    thread.start_new_thread(loop1,())
    '''
    sleep用来暂停主线程，否则两个newthread都不会执行；在不知道newthread何时执行完的情况下，这种同步机制并不可靠；
    '''
    sleep(6)
    print 'all DONE at:',ctime()
if __name__ == '__main__':
    main()