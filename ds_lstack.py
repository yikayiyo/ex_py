# -*- ecoding:utf-8 -*-
"""功能：栈的链接表实现，3154欢迎回来"""


class LNode:
    def __init__(self, elem, next_):
        self.elem = elem
        self.next = next_


class StackUnderflow(ValueError):
    pass


class LStack:
    def __init__(self):
        self._top = None

    def is_empty(self):
        return self._top is None

    def top(self):
        if self._top is None:
            raise StackUnderflow("in LStack.top()")
        return self._top.elem

    # 入栈操作：先是元素指向栈顶，然后栈顶指针指向插入元素
    def push(self, elem):
        l = LNode(elem, self._top)
        self._top = l

    def pop(self):
        if self._top is None:
            raise StackUnderflow("in LStack.pop()")
        p = self._top
        self._top = p.next
        return p.elem

    def printall(self):
        while not self.is_empty():
            print (self.pop())


if __name__ == '__main__':
    s = LStack()
    s.push(1)
    s.push(2)
    s.pop()
    s.push(3)
    s.printall()
    # s.pop()
    s.push(4)
    s.push(5)
    s.printall()
