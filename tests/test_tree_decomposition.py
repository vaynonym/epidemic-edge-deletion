import src.algorithms.tree_decomposition as td
import src.algorithms.nice_tree_decomposition as ntd
import networkx as nx
import matplotlib.pyplot as mpl

def test_create_introduce_node():

    test_td = td.Tree_Decomposer(nx.Graph())

    test_graph = nx.DiGraph()
    test_root = frozenset({1,2,3,4,5})
    nodes_dict = {}
    nodes_dict[frozenset(test_root)] = ntd.Nice_Tree_Node(test_root)
    nodes_dict[frozenset({1,2})] = ntd.Nice_Tree_Node({1,2})
    nodes_dict[frozenset({3})] = ntd.Nice_Tree_Node({3})
    nodes_dict[frozenset({1,3})] = ntd.Nice_Tree_Node({1,3})
    nodes_dict[frozenset({2,3})] = ntd.Nice_Tree_Node({2,3})
    nodes_dict[frozenset({1,2,3})] = ntd.Nice_Tree_Node({1,2,3})
    nodes_dict[frozenset({4,5})] = ntd.Nice_Tree_Node({4,5})

    # test 1
    # checks if the function creates the correct introduce nodes for
    # more than 1 child and more than 1 vertex in parent that is neither of the children
    test_graph.clear()

    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({2,3})])

    test_td.graph = test_graph
    test_td.graph_root = nodes_dict[frozenset(test_root)]

    test_td.create_introduce_nodes(nodes_dict[frozenset(test_root)])

    child = list(test_td.graph.successors(nodes_dict[frozenset(test_root)]))[0]
    assert not (child.bag == {1,2,3,4} or child.bag == {1,2,3,5}), "child has not specified bag"
    
    child_child = list(test_td.graph.successors(child))[0]
    assert not child_child == list(test_td.graph.successors(child))[0] == {1,2,3}, "child_child has not specified bag"

    # test 2
    # checks if the function correctly creates no introduce nodes for
    # 1 child and no vertex in parent that is neither of the children
    test_graph.clear()
    test_td.graph_root = nodes_dict[frozenset(test_root)]
    
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({4,5})])

    test_td.graph = test_graph
    
    test_td.create_introduce_nodes(nodes_dict[frozenset(test_root)])

    assert test_td.graph==test_graph, "should create no node or edge "
    
def test_create_join_node():

    test_td = td.Tree_Decomposer(nx.Graph())

    test_graph = nx.DiGraph()
    test_root = frozenset({1,2,3,4,5})
    nodes_dict = {}
    nodes_dict[frozenset(test_root)] = ntd.Nice_Tree_Node(test_root)
    nodes_dict[frozenset({1,2})] = ntd.Nice_Tree_Node({1,2})
    nodes_dict[frozenset({3})] = ntd.Nice_Tree_Node({3})
    nodes_dict[frozenset({1,3})] = ntd.Nice_Tree_Node({1,3})
    nodes_dict[frozenset({2,3})] = ntd.Nice_Tree_Node({2,3})
    nodes_dict[frozenset({1,2,3})] = ntd.Nice_Tree_Node({1,2,3})
    nodes_dict[frozenset({4,5})] = ntd.Nice_Tree_Node({4,5})

    # test 1
    # checks if the function creates the correct join nodes andfor
    # more than 1 child and more than 1 vertex in parent that is neither of the children
    test_graph.clear()

    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({2,3})])

    test_td.graph = test_graph
    test_td.graph_root = nodes_dict[frozenset(test_root)]

    test_td.create_join_node(nodes_dict[frozenset(test_root)])


def test_generate_partitions_with_2_blocks():
    test_set = {1,2,3}

    class test_class:
        def __init__(self):
            self.lint=10
    
    test_object1 = test_class()
    test_object2 = test_class()
    test_object3 = test_class()

    test_object1.lint = 10
    test_object2.lint = 11
    test_object3.lint = 12

    test_set_with_objects = {test_object1,test_object2,test_object3}

    test_td = td.Tree_Decomposer(nx.Graph())

    assert test_td.generate_partitions_with_2_blocks(test_set), ""
    assert test_td.generate_partitions_with_2_blocks(test_set_with_objects), ""

def test_create_forget_node():
    
    test_td = td.Tree_Decomposer(nx.Graph())

    test_graph = nx.DiGraph()
    test_root = frozenset({1,2,3,4,5})
    nodes_dict = {}
    nodes_dict[frozenset(test_root)] = ntd.Nice_Tree_Node(test_root)
    nodes_dict[frozenset({1,2})] = ntd.Nice_Tree_Node({1,2})
    nodes_dict[frozenset({3})] = ntd.Nice_Tree_Node({3})
    nodes_dict[frozenset({1,3})] = ntd.Nice_Tree_Node({1,3})
    nodes_dict[frozenset({2,3})] = ntd.Nice_Tree_Node({2,3})
    nodes_dict[frozenset({1,2,3})] = ntd.Nice_Tree_Node({1,2,3})
    nodes_dict[frozenset({4,5})] = ntd.Nice_Tree_Node({4,5})

    # test 1
    # checks if the function creates the correct forget nodes for
    # more than 1 child and more than 1 vertex in parent that is neither of the children
    test_graph.clear()

    test_graph.add_edge(nodes_dict[frozenset({1,2,3})],nodes_dict[frozenset(test_root)])

    test_td.graph = test_graph
    test_td.graph_root = nodes_dict[frozenset({1,2,3})]

    test_td.create_forget_nodes(nodes_dict[frozenset({1,2,3})])

    child = list(test_td.graph.successors(nodes_dict[frozenset({1,2,3})]))[0]
    assert not child == nodes_dict[frozenset(test_root)], "this child should be removed"
    assert (child.bag == {1,2,3,4} or child.bag == {1,2,3,5}), "child has not specified bag"
    
    child_child = list(test_td.graph.successors(child))[0]
    assert child_child == nodes_dict[frozenset(test_root)], "last node should be child"
    assert child_child.bag == {1,2,3,4,5}, "child_child has not specified bag"
