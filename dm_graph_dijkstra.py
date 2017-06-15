# -*- ecoding:utf-8 -*-
'''图,狄克斯特拉算法求最短路径,路径权值为正
'''


# 图

def find_lowest_cost_node(costs):
    lowest_cost = float("inf")
    lowest_cost_node = None
    for node in costs:
        cost = costs[node]
        # 选择开销最低且未被处理过的节点
        if cost < lowest_cost and node not in processed:
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node


def init():
    graph = {}
    graph["start"] = {}
    graph["start"]["a"] = 1
    graph["start"]["b"] = 2
    graph["a"] = {}
    graph["a"]["d"] = 3
    graph["a"]["c"] = 7
    graph["b"] = {}
    graph["b"]["d"] = 8
    graph["b"]["c"] = 5
    graph["d"] = {}
    graph["c"] = {}
    graph["c"]["fin"] = 3


    graph["fin"] = {}
    # 开销表
    infinity = float("inf")
    costs = {}
    costs["a"] = 1
    costs["b"] = 2
    costs["c"] = infinity
    costs["d"] = infinity
    costs["fin"] = infinity
    # 存储父节点
    parents = {}
    parents["a"] = "start"
    parents["b"] = "start"
    parents["c"] = None
    parents["d"] = None
    parents["fin"] = None
    # 处理过的节点
    processed = []
    return graph, costs, parents, processed


graph, costs, parents, processed = init()
node = find_lowest_cost_node(costs)
while node is not None:
    cost = costs[node]
    neighbors = graph[node]
    # 遍历当前节点的所有邻居
    for n in neighbors.keys():
        new_cost = cost + neighbors[n]
        # 经当前节点前往该邻居更近时，更新该邻居的开销和父节点
        if costs[n] > new_cost:
            costs[n] = new_cost
            parents[n] = node
            # 标记为已处理
    processed.append(node)
    node = find_lowest_cost_node(costs)


str=["fin"]
while parents["fin"]!="start":
    str.append(parents["fin"])
    tmp = parents["fin"]
    parents["fin"] = parents[tmp]
str.append(parents["fin"])
print '最短路径为：'
print "->".join(str[::-1])
print '长度为：'+costs["fin"].__str__()


