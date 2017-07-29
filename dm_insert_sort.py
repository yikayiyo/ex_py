# -*- ecoding:utf-8 -*-
'''插入排序'''


def insert_sort(lists):
    # 插入排序
    count = len(lists)
    cnt = 0
    for i in range(1, count):
        key = lists[i]
        j = i - 1
        while j >= 0:#边界条件
            cnt += 1
            # 这种写法在找到key的位置后,还会和前面的值比较.cnt==10
            if lists[j] > key:
                lists[j + 1] = lists[j]
                lists[j] = key
            j -= 1

            # 2.发现小于等于key后不再比较,因为前面的值已经有序了.cnt==8,最后的9没有和2,3比较.
            # if lists[j] > key:
            #     lists[j + 1] = lists[j]
            #     lists[j] = key
            #     j -= 1
            # else:
            #     break
    print cnt
    return lists

def insert_sort2(nums):
    count = len(nums)
    cnt=0
    for i in range(1,count):
        key=nums[i]
        while i-1 >= 0 and nums[i-1] > key:
            cnt+=1
            nums[i] = nums[i-1]
            i-=1
        nums[i] = key
    print cnt
    return nums

if __name__ == '__main__':
    lt = [10, 5, 2, 3, 9]
    # lt = [2, 3, 5, 9, 10]
    print insert_sort(lt)
    # print insert_sort2(lt)
