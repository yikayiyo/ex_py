class ArrayStack():
    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self) == 0

    def pop(self):
        if self.is_empty():
            raise IndexError
        return self._data.pop()
    def push(self, e):
        self._data.append(e)

    def top(self):
        if self.is_empty():
            raise IndexError
        return self._data[-1]
    def __str__(self):
        return str(self._data)

if __name__ == '__main__':
    a = ArrayStack()
    print(a)
    a.push(5)
    a.push(4)
    print(a)
    print(a.top())
    a.pop()
    print(a)
