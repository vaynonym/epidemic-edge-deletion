import src.algorithms.algorithm as alg
import networkx as nx

def test_algorithm():
	algo = alg.Algorithm(nx.Graph())



def test_generate_partitions_of_bag_of_size():

    #assemble
    algo = alg.Algorithm(nx.Graph())
    
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
    algo = alg.Algorithm(nx.Graph())


    partition = alg.Partition([[1,2,6,4,5], [6,10,3], [9,7,11,12], [13,14,15,16]])
    max_size = 7
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
    

    
    

	


