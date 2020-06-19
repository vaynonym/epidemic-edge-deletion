import src.algorithms.algorithm as alg
import networkx as nx
import src.algorithms.nice_tree_decomposition as ntd

def test_algorithm():
	algo = alg.Algorithm(nx.Graph(), nx.Graph(), 0, 0)

def test_hashing():
	sigma1 = (alg.Partition([alg.Block([1,2]), alg.Block([3])]), alg.Function({alg.Block([1]): 3}))
	sigma2 = (alg.Partition([alg.Block([1,2]), alg.Block([8])]), alg.Function({alg.Block([1]): 3}))
	sigma3 = (alg.Partition([alg.Block([1,2]), alg.Block([3])]), alg.Function({alg.Block([1]): 3}))

	d = dict()
	d[sigma1] = 8
	d[sigma2] = 3

	assert d[sigma3] == 8

def test_generate_partitions_of_bag_of_size():

    #assemble
    algo = alg.Algorithm(nx.Graph(), nx.Graph(), 0, 0)
    
    max_size = 5
    bag = set([1,2,3,4,5,6,7])

    # act
    result = algo.generate_partitions_of_bag_of_size(bag, max_size)

    # assert
    assert len(result) > 0
    for partition1 in result:
        for block1 in partition1.blocks:
            assert len(block1) > 0, "A block should be nonempty" 
            assert len(block1) <= max_size, "A block should have size smaller than the max_size"
            for block2 in partition1.blocks:
                if(not block1 == block2):
                    assert len(set(block1) & set(block2)) == 0, "The intersection of every block should be the empty set"
        
        
        flattened_blocks = [node for block in partition1.blocks for node in block]
        assert set(flattened_blocks) == bag, "The union of all blocks should be the bag"

        for partition2 in result:
            if(not partition1 == partition2):
                assert not partition1.blocks == partition2.blocks, "Every partition should be unique"
            
def test_generate_all_functions_of_partition():
    # assemble
    max_size = 7
    algo = alg.Algorithm(nx.Graph(), nx.Graph(), max_size, 0)


    partition = alg.Partition([alg.Block([1,2,6,4,5]), alg.Block([6,10,3]), alg.Block([9,7,11,12]), alg.Block([13,14,15,16])])
    # act
    result = algo.generate_all_functions_from_partition_to_range(partition, max_size)

    for function1 in result:
        for block in partition.blocks:
            assert block in function1.dictionary, "Every block needs to be assigned a value"    
        for key1 in function1.dictionary:
            assert function1.dictionary[key1] >= len(key1.node_list), "function(X) >= |X|"
            assert function1.dictionary[key1] <= max_size, "function(X) <= h"
        for function2 in result:
            if(not function1 == function2):
                functions_are_unique = False
                for key2 in function2.dictionary:
                    assert key2 in function1.dictionary, "every function should have the same keys"
                    if(not function1.dictionary[key2] == function2.dictionary[key2]):
                        functions_are_unique = True
                assert functions_are_unique, "Functions should be unique"
    
def test_find_component_signatures_of_leaf_nodes():
    # assemble
    nodes = [1, 2, 3, 4, 5, 6]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        [[node1, node2] for node2 in nodes for node1 in nodes if not node1 == node2]
    )
    nice_tree_decomposition = nx.Graph()
    h = 3
    k = 6
    algo = alg.Algorithm(graph, nice_tree_decomposition, h, k)
    
    bag = ntd.Nice_Tree_Node([2, 3, 4, 5])

    #act
    result = algo.find_component_signatures_of_leaf_nodes(bag, bag.bag)

    # assert (lol)
    #for key, del_value in result.items():
    #    print(del_value)



def test_find_component_signatures_of_join_nodes():
    # assemble
    nodes = [1, 2, 3, 4, 5, 6]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        [[node1, node2] for node2 in nodes for node1 in nodes if not node1 == node2]
    )
    nice_tree_decomposition = nx.Graph()

    h = 3
    k = 6
    algo = alg.Algorithm(graph, nice_tree_decomposition, h, k)
    
    TD_join_node = ntd.Nice_Tree_Node([2, 3, 4, 5])

    TD_child_node_1 = ntd.Nice_Tree_Node([2, 3, 4, 5])
    TD_child_node_1_del_k =  algo.find_component_signatures_of_leaf_nodes(TD_child_node_1, TD_child_node_1.bag)

    TD_child_node_2 = ntd.Nice_Tree_Node([2, 3, 4, 5])
    TD_child_node_2_del_k =  algo.find_component_signatures_of_leaf_nodes(TD_child_node_2, TD_child_node_2.bag)


    # merge both dictionaries
    TD_child_node_1_del_k.update(TD_child_node_2_del_k)
    del_values_child = TD_child_node_1_del_k
    #print(del_values_child)


    # act
    result = algo.find_component_signatures_of_join_nodes(TD_join_node, TD_join_node.bag, TD_child_node_1, TD_child_node_2, del_values_child)

    # assert (lol)
    #for key, del_value in result.items():
    #    print(del_value)










    



    
    

	


