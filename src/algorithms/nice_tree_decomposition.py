import networkx as nx

class Nice_Tree_Decomposition:
    def __init__(self, niceTD, treewidth):
        self.graph = niceTD
        self.treewidth = treewidth

    def successor(self, node):
        return self.graph.successor(node)

class Nice_Tree_Node:

    JOIN = "join"
    LEAF = "leaf"
    FORGET = "forget"
    INTRODUCE = "introduce"
    def __init__(self, bag, node_type=""):
        self.bag = set(bag)
        self.node_type = node_type
