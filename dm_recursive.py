#-*- ecoding:utf-8 -*-
'''递归练习'''
def mysum(list):
    #求和
    if (len(list) == 1):
        return list[0]
    else:
        return list[0] + mysum(list[1:])

def larger(a,b):
    if a>b:
        return a
    else:
        return b

def mymax(list):
    #找出列表中最大的数字
    if (len(list) == 2):
        return larger(list[0],list[1])
    else:
        return larger(list[0],mymax(list[1:]))

def mycount(list):
    #计算列表包含的元素数
    count=0
    if (len(list) == 1):
        return 1
    else:
        return 1 + mycount(list[1:])

def binary_search_rec(list,item,low,high):
    #递归二分查找
    if(low<=high):
        mid=(low+high)/2
        if(item==list[mid]):
            return mid
        elif item>list[mid]:
            return binary_search_rec(list,item,mid+1,high)
        else:
            return binary_search_rec(list,item,low,mid-1)
    return None

if __name__ == '__main__':
    lt=[1,2,3,4,5,6,7,8,9]
    print mysum(lt)
    print mymax(lt)
    print mycount(lt)
    print binary_search_rec(lt,-1,0,8)
    print binary_search_rec(lt,9,0,8)