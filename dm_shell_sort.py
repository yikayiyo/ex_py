# -*- ecoding:utf-8 -*-
'''希尔排序'''


def shell_sort(nums):
    count = len(nums)
    dist = 1
    while dist < count / 3: dist *= 3
    while dist > 0:
        for i in range(dist,count):
            key=nums[i]
            while i-dist >= 0 and nums[i-dist] > key:
                nums[i] = nums[i-dist]
                i-=dist
            nums[i] = key
        dist/=3
    return nums


if __name__ == '__main__':
    lt = [10, 5, 2, 3, 9]
    print shell_sort(lt)
