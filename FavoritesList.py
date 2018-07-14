import PositionalList
# from PositionalList import *

class FavoritesList:
    '''记录一些经常访问的元素'''

    class _Item:
        __slots__ = '_value', '_count'

        def __init__(self, e):
            self._value = e
            self._count = 0

    # -----------------------公有类
    def __init__(self):
        self._data = PositionalList.PositionalList()
        # 第二种导入方法用
        # self._data = PositionalList()

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def access(self, e):
        p = self._find_position(e)
        if p is None:
            p = self._data.add_last(self._Item(e))
        p.element()._count += 1
        self._move_up(p)

    def remove(self, e):
        p = self._find_position(e)
        if p is not None:
            self._data.delete(p)

    def top(self, k):
        if not 1 <= k <= len(self):
            raise ValueError('Illegal value for k')
        walk = self._data.first()
        for j in range(k):
            item = walk.element()
            yield item._value
            walk = self._data.after(walk)

    # -----------------------非公有工具类
    def _find_position(self, e):
        '''找到e返回其位置'''
        walk = self._data.first()
        while walk is not None and walk.element()._value != e:
            walk = self._data.after(walk)
        return walk

    def _move_up(self, p):
        '''根据访问次数移动p处的item'''
        if p != self._data.first():
            cnt = p.element()._count
            walk = self._data.before(p)
            if cnt > walk.element()._count:  # must shift forward
                while walk != self._data.first() and cnt > self._data.before(walk).element()._count:
                    walk = self._data.before(walk)
                self._data.add_before(walk, self._data.delete(p))  # delete/reinsert

if __name__ == '__main__':
    fl = FavoritesList()
    assert len(fl)==0
    fl.access(4)
    fl.access(4)
    fl.access(4)
    fl.access(4)
    fl.access(1)
    fl.access(1)
    fl.access(1)
    fl.access(2)
    fl.access(2)
    fl.access(3)
    assert list(fl.top(2))==[4,1]