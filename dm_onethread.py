#-*- ecoding:utf-8 -*-
'''该脚本在一个单线程中连续执行两个循环，一个在另一个之后紧接执行。总耗时为两个循环时间的累加'''
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
    loop0()
    loop1()
    print 'all DONE at:',ctime()
if __name__ == '__main__':
    main()