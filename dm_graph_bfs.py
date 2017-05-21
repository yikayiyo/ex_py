#-*- ecoding:utf-8 -*-
'''图,广度遍历，在非加权图中查找最短路径'''
from collections import deque

def person_is_seller(name):
    return name[-1] == 'm'
#图
graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []
#借助队列实现BFS
def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []#这个数组用于记录检查过的人
    while search_queue:
        person = search_queue.popleft()
        if person not in searched:#避免重复检查
            if person_is_seller(person):#随意设计的函数
                print person + " is a mango seller!"
                return True
            else:
                search_queue += graph[person]
                searched.append(person)
    return False

if __name__ == '__main__':
    search("you")
