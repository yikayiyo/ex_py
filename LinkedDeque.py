from DoublyLinkedBase import *

class LinkedDeque(DoublyLinkedBase):
    #
    # def __init__(self):
    #     super.__init__()
    def first(self):
        if self.is_empty():
            raise IndexError
        return self._header._next._elem

    def last(self):
        if self.is_empty():
            raise IndexError
        return self._trailer._prev._elem

    def insert_first(self,e):
        self._insert_between(e,self._header,self._header._next)

    def insert_last(self,e):
        self._insert_between(e,self._trailer._prev,self._trailer)

    def delete_first(self):
        if self.is_empty():
            raise IndexError
        self._delete_node(self._header._next)

    def delete_last(self):
        if self.is_empty():
            raise IndexError
        self._delete_node(self._trailer._prev)

if __name__ == '__main__':
    lq = LinkedDeque()
    print(lq)
    lq.insert_first(1)
    lq.insert_first(2)
    lq.insert_first(3)
    lq.insert_first(4)
    assert lq.last() == 1
    lq.insert_last(5)
    assert lq.last() == 5
    lq.delete_first()
    lq.delete_last()
    assert lq._size == 3
    print(lq)
