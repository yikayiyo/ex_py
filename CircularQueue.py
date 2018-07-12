class _Node():
    def __init__(self,elem,next=None):
        self._elem = elem
        self._next = next

class CircularQueue():
    '''
    利用循环链表实现队列
    '''
    def __init__(self):
        self._tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size==0

    def fist(self):
        '''
        队首出队
        :return:
        '''
        if self.is_empty():
            raise IndexError
        # 只有一个节点时也可以这样
        head = self._tail._next
        return head._elem

    def dequeue(self):
        if self.is_empty():
            raise IndexError
        oldhead = self._tail._next
        if self._size==1:
            self._tail = None
        else:
            self._tail = oldhead._next
        self._size -= 1
        return oldhead._elem

    def enqueue(self,e):
        newest = _Node(e)
        #第一次入队
        if self.is_empty():
            newest._next = newest
        else:
            newest._next = self._tail._next
            self._tail._next = newest
        #最后都要更新tail
        self._tail = newest
        self._size += 1

    def rotate(self):
        # 队首元素放到队尾
        if self._size > 0:
            self._tail = self._tail._next

    def __str__(self):
        s=[]
        p = self._tail
        while self._size:
            s.append(str(p._next._elem))
            p = p._next
            self._size -= 1
        return '-'.join(s)

if __name__ == '__main__':
    cq = CircularQueue()
    print(cq.is_empty())
    cq.enqueue(5)
    cq.enqueue(4)
    cq.enqueue(3)
    print(len(cq))
    print(cq)