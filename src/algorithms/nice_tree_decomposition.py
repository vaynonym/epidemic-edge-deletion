import networkx as nx

class Nice_Tree_Decomposition:
    def __init__(self, niceTD):
        self.graph = niceTD
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                self.root = node
                break
        
    def successors(self, node):
        return self.graph.successors(node)

    def predecessors(self, node):
        return self.graph.predecessors(node)
    
    def find_leafs(self):
        return [x for x in self.graph.nodes() if self.graph.out_degree(x)==0 and self.graph.in_degree(x)==1]

class Nice_Tree_Node:

    JOIN = "join"
    LEAF = "leaf"
    FORGET = "forget"
    INTRODUCE = "introduce"
    def __init__(self, bag, node_type=""):
        self.bag = set(bag)
        self.node_type = node_type
