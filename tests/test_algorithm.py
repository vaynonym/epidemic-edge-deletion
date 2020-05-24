import src.algorithms.algorithm as alg
import networkx as nx

def test_algorithm():
	algo = alg.Algorithm(nx.Graph())



def test_generate_partitions_of_bag_of_size():

    #assemble
    algo = alg.Algorithm(nx.Graph())
    
    max_size = 8
    bag = set([1,2,3,4,5,6,7, 8])

    # act
    result = algo.generate_partitions_of_bag_of_size(bag, max_size)
    for partition in result:
        print(partition.blocks)

    # assert

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
            

    

	


