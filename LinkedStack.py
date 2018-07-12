class _Node():
    def __init__(self, elem, next=None):
        self._elem = elem
        self._next = next

class LinkedStack():

    def __init__(self):
        self._head = None
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self.__len__()==0

    def push(self,e):
        self._head = _Node(e,next=self._head)
        self._size += 1

    def top(self):
        if self.is_empty():
            raise IndexError
        return self._head._elem

    def pop(self):
        if self.is_empty():
            raise IndexError
        answer = self._head._elem
        self._head = self._head._next
        self._size -= 1
        return answer

    def __str__(self):
        p = self._head
        if not p:return 'NULL'
        s=[]
        while p:
            s.append(str(p._elem))
            p = p._next
        return '-'.join(s)

if __name__ == '__main__':
    ls = LinkedStack()
    print(ls)
    ls.push(5)
    ls.push(4)
    ls.push(3)
    ls.push(2)
    ls.push(1)
    print(ls)
    print(ls.top())
    ls.pop()
    ls.pop()
    print(ls)