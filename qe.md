```py
# 问题：虽然内容都是１，但是v1，v2是两个不同的Vertex实例
# 节点的插入是多余的
from graph import Graph
g = Graph(directed=True)
v1 = g.insert_vertex(1)
v2 = g.insert_vertex(1)
id(v1)
Out[6]: 139793589469136
id(v2)
Out[7]: 139793580593336
```
