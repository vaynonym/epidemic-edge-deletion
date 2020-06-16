import copy
import math

class Algorithm:
	
	def __init__(self, graph, nice_tree_decomposition, h, k):
		self.graph = graph
		self.h = h
		self.k = k
		#self.nice_tree_decomposition = nice_tree_decomposition
		#self.nodes_to_be_calculated = nice_tree_decomposition.find_leafs()
		#self.root = nice_tree_decomposition.root
		#self.root_set = set(nice_tree_decomposition.root)


	def execute(self):

		#nodes_to_be_calculated = nice_tree_decomposition.find_leafs()
		#root_set = set(nice_tree_decomposition.root)

		

		#while(not nodes_to_be_calculated == root_set)
		#	new_nodes_to_be_calculated = set(nodes_to_be_calculated)
		#	for node in nodes_to_be_calculated:
		#		if(self.can_node_be_calculated(node)):
		#			self.component_signatures[node] = self.calculate_component_signature_of_node(node)
		#			# update nodes_to_be_calculated
		#			new_nodes_to_be_calculated.remove(node)
		#			if(not node == root)
		#				new_nodes_to_be_calculated.add(nice_tree_decomposition.graph.predecessor(node))
		#	nodes_to_be_calculated = new_nodes_to_be_calculated
		
		return True

	#def update_nodes_to_be_calculated(self, node, new_nodes_to_be_calculated):
	#	new_nodes_to_be_calculated.remove(node)
	#	if(not node == root)
	#		new_nodes_to_be_calculated.add(nice_tree_decomposition.graph.predecessor(node))

	#def can_node_be_calculated(self, node):
	#	can_be_calculated = 0
	#	successors = node.get_successors
	#	for successor in successors:
	#		if(successor in self.component_signatures):
	#			can_be_calculated = can_be_calculated + 1
	#	
	#	return can_be_calculated == len(successors)
			


	def generate_possible_component_states_of_bag(self, bag, h):
		states = set()
		all_partitions = self.generate_partitions_of_bag_of_size(bag, h)
		for partition in all_partitions:
			all_functions = self.generate_all_functions_from_partition_to_range(partition, h)
			for function in all_functions:
				states.add( (partition, function))

		return states
			
	def generate_partitions_of_bag_of_size(self, bag, size):
		partitions = set()
		initial_partition = Partition([])
		for node in bag:
			initial_partition.blocks.append([])

		partitions.add(initial_partition)

		for node in bag:
			new_partitions = set()
			
			for partition in partitions:
				# iterate over each block, using a for loop over an index because
				# we need the index for the blocks in the new partition we create
				for i in range(len(partition.blocks)):
					if len(partition.blocks[i]) < size:
						
						new_partition = partition.get_copy()
						
						new_partition.blocks[i].append(node)

						# check if identical partition already exists
						does_identical_partition_exist = False
						for existing_partition in new_partitions:
							is_a_block_different = False
							for block in existing_partition.blocks:
								if not block in new_partition.blocks:
									is_a_block_different = True
									break
							if not is_a_block_different:
								does_identical_partition_exist = True
								break

						if(not does_identical_partition_exist):
							new_partitions.add(new_partition)
			
			partitions = new_partitions

		for partition in partitions:
			blocks_to_remove = []
			for block in partition.blocks:
				if block == []:
					blocks_to_remove.append(block)
			
			for block in blocks_to_remove:
				partition.blocks.remove(block)

		return partitions

	# a function here is just a dictionary that maps its unique inputs values to output values
	def generate_all_functions_from_partition_to_range(self, partition, h):
		all_functions = set()
		all_functions.add(Function({})) # initial set of undefined functions used to generate the rest

		new_block_list = []
		for block in partition.blocks:
			new_block_list.append(Block(block)) # blocks need to be hashable now so we create a wrapper object
		partition.blocks = new_block_list
	
		
		# partially define function by setting the mapping for each block
		for block in partition.blocks:

			# for each possible value that a block could be mapped to
			new_all_functions = []
			for codomain_value in range(1, h+1):
				# if that value is legal
				if codomain_value >= len(block.node_list):
					# then for each function that already exists
					for function in all_functions:
						# create a new function with the only difference being
						# that the new block now has a mapping in that function

						new_function = Function(dict(function.dictionary))
						new_function.dictionary[block] = codomain_value
						new_all_functions.append(new_function)
			
			all_functions = new_all_functions

		return all_functions

	# Algorithm 2
	def find_component_signatures_of_leaf_nodes(self, leaf_node, bag):
		del_values = dict()
		all_states = self.generate_possible_component_states_of_bag(bag, self.h)
		for state in all_states:
			potential_edges_to_remove = set()
			induced_subgraph = self.graph.subgraph(list(bag))

			potential_edges_to_remove.update(self.edges_connecting_blocks_in_partition(induced_subgraph.edges, state))
			
			# for now I'm saving the del-values themselves because we'll probably need them
			# for the algorithm as suggested by the paper, the second value of the tuple is what we need
			if(len(potential_edges_to_remove) <= self.k):
				del_values[(leaf_node, state)] = (potential_edges_to_remove, len(potential_edges_to_remove))
			else:
				# in this case, there should be no need to save the actual set of edges
				# so I'm leaving it as an empty set for performance reasons
				del_values[(leaf_node, state)] = (set(), math.inf)
		
		return del_values

	# takes the edges of the induced subgraph and the current state
	# returns set of edges
	def	edges_connecting_blocks_in_partition(self, induced_subgraph_edges, state):
		result = set()
		for (u,w) in induced_subgraph_edges:
			block_containing_u = Block(set())
			block_containing_w = Block(set())
			for block in state[0].blocks:
				if(u in block.node_list):
					block_containing_u = block
				if(w in block.node_list):
					block_containing_w = block
				
				if(block_containing_u.node_list != set() and block_containing_w.node_list != set()):
					break
				
			if(block_containing_u != block_containing_w):
				result.add((u,w))
		return result
				

			



# We need a wrapper class in order to have hashable lists for the set of Partitions
class Block:
	def __init__(self, node_list):
		self.node_list = node_list

	def __getitem__(self, key):
		return self.node_list[key]

class Partition:
	def __init__(self, blocks):
		self.blocks = blocks

	def __repr__(self):
		return repr(self.blocks)

	def get_copy(self):
		new_block_list = []
		for block in self.blocks:
			new_block_list.append(list(block))
		return Partition(new_block_list)

	def __getitem__(self, key):
		return self.blocks[key]

class Function:
	def __init__(self, dictionary):
		self.dictionary = dictionary

	def __getitem__(self, key):
		return self.dictionary[key]





