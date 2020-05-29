import networkx as nx

class Nice_Tree_Decomposition:
    def __init__(self, niceTD, treewidth):
        self.graph = niceTD
        self.treewidth = treewidth

    def successor(self, node):
        return self.graph.successor(node)

class Nice_Tree_Node:

    def __init__(self, bag):
        self.bag=set(bag)
        self.node_type=""
