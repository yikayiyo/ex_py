#-*- ecoding:utf-8 -*-
'''快速排序'''
def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less=[i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

if __name__ == '__main__':
    lt=[10,5,2,3,9]
    print quicksort(lt)