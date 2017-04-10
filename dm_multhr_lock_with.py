#-*- ecoding:utf-8 -*-
'''锁和上下文管理'''
from atexit import register
from random import randrange
from threading import Thread,Lock,current_thread
from time import ctime,sleep
class CleanOutputSet(set):
    def __str__(self):
        return ', '.join(x for x in self)
#全局变量,3~6个线程,2~4秒的休眠时间
lock=Lock()
loops=(randrange(2,5) for x in xrange(randrange(3,7)))
remaining=CleanOutputSet()

def loop(nsec):
    myname=current_thread().name
    #with是个好东西-0-
    with lock:
        remaining.add(myname)
        print '[%s] started %s'%(ctime(),myname)
    sleep(nsec)
    with lock:
        remaining.remove(myname)
        print '[%s] completed %s (%d secs)' %(ctime(),myname,nsec)
        print ' (remaining: %s)'%(remaining or 'NONE')

def main():
    for pause in loops:
        Thread(target=loop,args=(pause,)).start()

@register
def atexit():
    print 'ALL DONE AT:',ctime()

if __name__ == '__main__':
    main()





