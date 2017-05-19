# -*- ecoding:utf-8 -*-
'''二分查找'''


def binary_search(list, item):
    low = 0
    high = len(list) - 1
    while low <= high:
        mid = (low + high) / 2
        guess = list[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None

if __name__ == '__main__':
    list=[1,2,3,4,7]
    print binary_search(list,-1)
    print binary_search(list,3)