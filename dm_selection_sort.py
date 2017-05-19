#-*- ecoding:utf-8 -*-
'''选择排序'''
def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1,len(arr)):
        #每次都找最小值
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr=[]
    for i in range(len(arr)):
        # 最小值放入新数组
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))

    return newArr

if __name__ == '__main__':
    print selectionSort([5,3,6,2,10,9])
