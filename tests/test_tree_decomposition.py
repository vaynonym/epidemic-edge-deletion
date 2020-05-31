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
    # more than 1 child and more than 1 vertex in parent that is in neither of the children
    test_graph.clear()

    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({2,3})])

    test_td.graph = test_graph
    test_td.graph_root = nodes_dict[frozenset(test_root)]

    test_td.create_introduce_nodes(nodes_dict[frozenset(test_root)])

    child = list(test_td.graph.successors(nodes_dict[frozenset(test_root)]))[0]
    assert (child.bag == {1,2,3,4} or child.bag == {1,2,3,5}), "child has not specified bag"

    for node_deg in list(nx.degree(test_td.graph)):
        assert not node_deg[1]==0, "does create isolated node(s)"

    assert nx.number_of_selfloops(test_td.graph) == 0, "creates self loop(s)"
    assert nx.is_weakly_connected(test_td.graph), "not weakly connected"

    #-----------------------------------------------------------------------------------------
    test_td.graph.remove_edge(test_td.graph_root, child)

    test_td.graph.remove_edge(child, nodes_dict[frozenset({1,2})])
    test_td.graph.remove_edge(child, nodes_dict[frozenset({3})])
    test_td.graph.remove_edge(child, nodes_dict[frozenset({1,3})])
    test_td.graph.remove_edge(child, nodes_dict[frozenset({2,3})])

    assert nx.is_empty(test_td.graph), "creates additional edge(s) or does not remove specified edge(s)"

    # test 2
    # checks if the function correctly creates no introduce nodes for
    # 1 child and no vertex in parent that is not in the child
    test_graph.clear()
    test_td.graph_root = nodes_dict[frozenset(test_root)]
    
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({4,5})])

    test_td.graph = test_graph
    
    test_td.create_introduce_nodes(nodes_dict[frozenset(test_root)])
    
    for node_deg in list(nx.degree(test_td.graph)):
        assert not node_deg[1]==0, "does create isolated node(s)"

    assert nx.number_of_selfloops(test_td.graph) == 0, "creates self loop(s)"
    assert nx.is_weakly_connected(test_td.graph), "not weakly connected"

    #-----------------------------------------------------------------------------------------
    test_graph.remove_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2,3})])
    test_graph.remove_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({4,5})])

    assert nx.is_empty(test_td.graph), "creates additional edge(s) or does not remove specified edge(s)"

    # test 3
    # checks if the function correctly creates introduce nodes for
    # 1 child and 1+ vertex in parent that is not in the child

    test_graph.clear()
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2,3})])

    test_td.graph = test_graph
    
    test_td.create_introduce_nodes(nodes_dict[frozenset(test_root)])
    
    for node_deg in list(nx.degree(test_td.graph)):
        assert not node_deg[1]==0, "does create isolated node(s)"

    assert nx.number_of_selfloops(test_td.graph) == 0, "creates self loop(s)"
    assert nx.is_weakly_connected(test_td.graph), "not weakly connected"

    #-----------------------------------------------------------------------------------------
    test_td.graph.remove_edge(list(test_td.graph.successors(nodes_dict[frozenset(test_root)]))[0],nodes_dict[frozenset({1,2,3})])
    test_td.graph.remove_edge(nodes_dict[frozenset(test_root)],list(test_td.graph.successors(nodes_dict[frozenset(test_root)]))[0])

    assert nx.is_empty(test_td.graph), "creates additional edge(s) or does not remove specified edge(s)"

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
    # more than 2 child and more than 1 vertex in parent that is neither of the children
    test_graph.clear()

    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,2})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({1,3})])
    test_graph.add_edge(nodes_dict[frozenset(test_root)],nodes_dict[frozenset({2,3})])

    test_td.graph = test_graph
    test_td.graph_root = nodes_dict[frozenset(test_root)]

    test_td.create_join_node(nodes_dict[frozenset(test_root)])
    
    child_left = list(test_td.graph.successors(test_td.graph_root))[0]
    child_right = list(test_td.graph.successors(test_td.graph_root))[1]

    assert child_left.bag == test_td.graph_root.bag , "left child does not have same bag as join node"
    assert child_right.bag == test_td.graph_root.bag, "right child does not have same bag as join node"

    for node_deg in list(nx.degree(test_td.graph)):
        assert not node_deg[1]==0, "does create isolated node(s)"

    assert nx.number_of_selfloops(test_td.graph) == 0, "creates self loop(s)"
    assert nx.is_weakly_connected(test_td.graph), "not weakly connected"

    for child in [nodes_dict[frozenset({1,2})],nodes_dict[frozenset({3})],nodes_dict[frozenset({1,3})],nodes_dict[frozenset({2,3})]]:
        assert test_td.graph.has_edge(child_left, child) or test_td.graph.has_edge(child_right, child), "a child is left alone D:"
    #-----------------------------------------------------------------------------------------
        if test_td.graph.has_edge(child_left, child):
            test_td.graph.remove_edge(child_left,child)

        if test_td.graph.has_edge(child_right, child):
            test_td.graph.remove_edge(child_right,child)

    test_td.graph.remove_edge(test_td.graph_root, child_right)
    test_td.graph.remove_edge(test_td.graph_root, child_left)

    assert nx.is_empty(test_td.graph), "creates additional edge(s) or does not remove specified edge(s)"

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
    # 1 child and more than 1 vertex in parent that is not he child
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

    for node_deg in list(nx.degree(test_td.graph)):
        assert not node_deg[1]==0, "does create isolated node(s)"

    assert nx.number_of_selfloops(test_td.graph) == 0, "creates self loop(s)"
    assert nx.is_weakly_connected(test_td.graph), "not weakly connected"