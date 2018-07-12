class _Node():
    def __init__(self, elem,next=None):
        self._elem = elem
        self._next = next

class SinglyLinkedList():

    def __init__(self,head=None,tail=None):
        # 带头指针，尾指针，和链表大小
        self.head = head
        self.tail = tail
        self.size = 0

    def add_fist(self,e):
        newest = _Node(e)
        newest._next = self.head
        self.head = newest
        self.size += 1

    def add_last(self,e):
        newest = _Node(e)
        if not self.tail:
            self.tail = newest
            self.head = newest
        elif self.tail:
            self.tail._next = newest
            self.tail = newest
        self.size += 1

    def remove_first(self):
        if self.head is None:
            raise IndexError
        else:
            self.head = self.head._next
            self.size -= 1

    def __str__(self):
        p = self.head
        if p is None:
            return 'NULL'
        s=[]
        while p is not None:
            s.append(str(p._elem))
            p = p._next
        return '-'.join(s)

    def remove_last(self):
        # 遍历找到tail的前一个位置
        pass
if __name__ == '__main__':
    l = SinglyLinkedList()
    # print(l)
    l.add_last(5)
    l.add_last(4)
    l.add_last(3)
    l.add_last(2)
    l.add_last(1)
    print(l)
    l.remove_first()
    print(l)
    l.add_fist(5)
    print(l)
