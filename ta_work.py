#-*- encoding:utf-8 -*-
'''统计同学们交的作业总数'''
import re
#获取学生名单，返回列表用于之后的写入，字典用于计数
def init_stu(filename):
    d = {}
    l=[]
    with open(filename) as f:
        for line in f.readlines():
            lt=line.strip().split('\t')
            if lt not in l:
                l.append(lt)
            # print line
            if lt[0] not in d:
                d[lt[0]]=0
    return l,d
#抓取网页上的学号，返回一个没有重复元素的list
def handle_html(filename):
    list=[]
    with open(filename) as f:
        for line in f.readlines():
            # print line
            res=re.search(r'11[0-9]+',line)
            if res and res not in list:
                list.append(res.group())
    return set(list)


def my_print(list):
    print len(list)
    if list:
        for item in list:
            print item

#根据list更新dic内容，存在则 +1 -0-
def counter(dic,list):
    for item in list:
        if item in dic.keys():
            dic[item]+=1
    return dic

#将学号、姓名、作业次数写入文件
def write_into_file(filename,list,stu):
    with open(filename,'w') as f:
        for item in list:
            id,name=item[0],item[1]
            item.append(str(stu[id]))
            f.writelines('\t'.join(item)+'\n')
    print 'work done.'

if __name__ == '__main__':
    listfile = 'stu_list'
    resfile='res.txt'
    list,stu=init_stu(listfile)
    #事先抓取8个页面并重命名
    for i in range(1,9):
        path='homework'+str(i)+'.html'
        l=handle_html(path)
        stu=counter(stu,l)
    # print stu
    write_into_file(resfile,list,stu)
