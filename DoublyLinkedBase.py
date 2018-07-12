class DoublyLinkedBase():
    '''双向链表'''

    class _Node():
        __slots__ = '_elem','_prev','_next'
        def __init__(self,elem,next=None,prev=None):
            '''
            节点定义
            :param elem:　对象引用
            :param next: 后向指针
            :param prev: 前向指针
            '''
            self._elem = elem
            self._next = next
            self._prev = prev

    def __init__(self):
        self._header = self._Node(None)
        self._trailer = self._Node(None)
        self._header._next = self._trailer
        self._trailer._prev = self._header
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size==0

    def insert_between(self,e,predecessor,successor):
        newest = self._Node(e,prev=predecessor,next=successor)
        predecessor._next = newest
        successor._prev = newest
        self._size += 1
        return newest

    def _delete_node(self,node):
        predecessor = node._prev
        successor = node._next
        predecessor._next = successor
        successor._prev = predecessor
        self._size -= 1
        # 记录被删除节点
        element  = node._elem
        # 两种写法都可以
        # del node
        node._prev = node._next = node._elem = None
        return element

    def __str__(self):
        if self.is_empty():
            return 'NULL'
        s=[]
        p = self._header._next
        # 这里一定要取临时变量，直接操作self.size会很惨
        size = self._size
        while size:
            s.append(str(p._elem))
            p = p._next
            size -=1
        return '-'.join(s)

if __name__ == '__main__':
    dl = DoublyLinkedBase()
    print(dl)
    node5 = dl.insert_between(5,dl._header,dl._trailer)
    print(dl._size)
    node6 = dl.insert_between(6,node5,dl._trailer)
    print(dl._size)
    node7 = dl.insert_between(7,node6,dl._trailer)
    print(dl._size)
    print(dl)
    node = dl._delete_node(node6)
    print(dl)