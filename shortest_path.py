class AdaptableHeapPriorityQueue():
    pass

def shortest_path_lengths(g, src):
    '''
    compute shortest-path distances from src to reachable vertices of g
    :param g: Graph can be directed or undirected,but must be weighted
    :param src: start vertex
    :return: dictionary mapping each reachable vertex to its distance from src
    '''
    d = {} #临时路径长度
    cloud = {} # v：d[v]
    pq = AdaptableHeapPriorityQueue() #d[v]作为键的优先级队列，用于取最近的节点
    pqlocator = {} # 记录节点位置，用于更新节点信息

    #初始化d，起始节点为0，其它为无穷大
    for v in g.vertices():
        if v is src:
            d[v] = 0
        else:
            d[v] = float('inf')
        pqlocator[v] = pq.add(d[v], v)

    # 贪心迭代
    while not pq.is_empty():
        key, u = pq.remove_min() #取最近的节点
        cloud[u] = key           #加入cloud
        del pqlocator[u]
        for e in g.incident_edges(u): #所有的边(u,v)
            v = e.opposite(u)
            if v not in cloud:        #更新不在cloud中，和u挨着的d[v]
                wgt = e.element()
                if d[u] + wgt < d[v]:
                    d[v] = d[u] + wgt
                    pq.update(pqlocator[v], d[v], v)
    return cloud
