```
一个二维数组中,数字从左到右递增,从上到下也是递增.判断数组中是否有某个数字.
```
def find(matrix, rows, cols, number):
    found = False
    if matrix and rows and cols:
        row = 0
        col = cols - 1
        while row < rows and col >= 0:
            key = matrix[row * cols + col]
            if key == number:
                found = True
                break
            elif key > number:
                col -= 1
            else:
                row += 1
    return found

if __name__ == '__main__':
    matrix = [1, 2, 8, 9, 2, 4, 9, 12, 4, 7, 10, 13, 6, 8, 11, 15]
    rows, cols = 4, 4
    # 测试用例
    # 数组中存在值的情况
    assert find(matrix, rows, cols, 1) == True, '1'
    assert find(matrix, rows, cols, 15) == True, '1'
    assert find(matrix, rows, cols, 7) == True, '1'
    # 数组中不存在值的情况
    assert find(matrix,rows,cols,20)==False,'0'
    assert find(matrix,rows,cols,5)==False,'0'
    assert find(matrix,rows,cols,0)==False,'0'
    #数组为空,下标越界的情况
    assert find([],rows,cols,13)==False,'0'
    assert find(matrix,-1,cols,13)==False,'0'
    assert find(matrix,rows,0,13)==False,'0'
