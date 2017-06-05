#-*- ecoding:utf-8 -*-
'''循环单链表'''
#节点定义
class LNode:
    def __init__(self,elem,next_=None):
        self.elem=elem
        self.next = next_
#链表定义
class LCList:
    def __init__(self):
        self._rear = None
    def is_empty(self):
        return self._rear is None
    #前端插入
    def prepend(self,elem):
        p=LNode(elem)
        if self._rear is None:
            p.next=p
            self._rear=p
        else:
            p.next=self._rear.next
            self._rear.next=p
    #后端插入,唯一区别是尾指针的移动
    def append(self,elem):
        self.prepend(elem)
        self._rear=self._rear.next
    # 前端弹出
    def pop(self):
        if self._rear is None:
            raise OverflowError
        p=self._rear.next
        if p is self._rear:
            self.rear=None
        else:
            self._rear.next=p.next
        return p.elem

    def printall(self):
        if self.is_empty():
            return
        p=self._rear.next
        while True:
            print p.elem
            if p is self._rear:
                break
            p=p.next
if __name__ == '__main__':
    lclist = LCList()
    for i in range(1,5):
        lclist.append(i)
    lclist.printall()
