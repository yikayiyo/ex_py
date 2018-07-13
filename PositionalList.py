from DoublyLinkedBase import *

class PositionalList(DoublyLinkedBase):
    # -------------------------- nested Position class --------------------------
    class Position():
        def __init__(self,container,node):
            self._container = container
            self._node = node
        def element(self):
            return self._node._elem
        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node
        def __ne__(self, other):
            return not(self==other)
    # ------------------------------- utility method -------------------------------
    def _validate(self, p):
        '''return positon's node,or raise error if invalid.'''
        if not isinstance(p, self.Position):
            raise TypeError('p must be proper Position type')
        if p._container is not self:
            raise ValueError('p does not belong to this container')
        if p._node._next is None:
            raise ValueError('p is no longer valid')
        return p._node

    def _make_position(self, node):
        '''给定节点引用返回位置'''
        if node is self._header or node is self._trailer:
            return None
        return self.Position(self,node)
    # ------------------------------- accessors -------------------------------
    def first(self):
        '''Return the first Position in the list (or None if list is empty).'''
        return self._make_position(self._header._next)

    def last(self):
        '''Return the last Position in the list (or None if list is empty).'''
        return self._make_position(self._trailer._next)

    def before(self,p):
        '''Return the Position just before Position p (or None if p is first).'''
        node = self._validate(p)
        return self._make_position(node._prev)

    def after(self,p):
        '''Return the Position just before Position p (or None if p is first).'''
        node = self._validate(p)
        return self._make_position(node._next)

    def __iter__(self):
        cusor = self.first()
        while cusor is not None:
            yield cusor.element()
            cusor = self.after(cusor)
    # ------------------------------- mutators -------------------------------
    # override inherited version to return Position, rather than Node
    def _insert_between(self,e,predecessor, successor):
        #覆盖父类方法，返回位置
        node = super()._insert_between(e, predecessor, successor)
        return self._make_position(node)

    def add_first(self,e):
        return self._insert_between(e,self._header,self._header._next)

    def add_last(self,e):
        '''在Ｌ最后插入'''
        return self._insert_between(e,self._trailer._prev,self._trailer)

    def add_before(self,p,e):
        original = self._validate(p)
        return self._insert_between(e,original._prev,original)

    def add_after(self,p,e):
        '''在位置ｐ之后插入'''
        original = self._validate(p)
        return self._insert_between(e,original,original._next)

    def delete(self,p):
        '''删除并返回位置p处的元素'''
        original = self._validate(p)
        return self._delete_node(original)

    def replace(self,p,e):
        '''用e替换p处的值，返回原始值'''
        original = self._validate(p)
        old_value = original._elem
        original._elem = e
        return old_value

if __name__ == '__main__':
    pl = PositionalList()
    assert pl.first() is None
    p5 = pl.add_first(5)
    p4 = pl.add_last(4)
    p2 = pl.add_last(2)
    assert len(pl)==3
    pl.add_after(p5,3)
    assert str(pl)=='5-3-4-2','not 5342'
    pl.replace(p5,6)
    # print(pl)
    assert str(pl) == '6-3-4-2', 'not 6342'