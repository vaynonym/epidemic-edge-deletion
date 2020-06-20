import copy
import math
import src.algorithms.nice_tree_decomposition as ntd
import multiprocessing
import threading
import sys

class Algorithm:
	
	def __init__(self, graph, nice_tree_decomposition, h, k):
		self.graph = graph
		self.h = h
		self.k = k
		self.nice_tree_decomposition = nice_tree_decomposition
		self.nodes_to_be_calculated = self.nice_tree_decomposition.find_leafs()
		self.root = nice_tree_decomposition.root
		self.component_signatures = dict()


	def execute(self):

	
		nodes_to_be_calculated = self.nice_tree_decomposition.find_leafs()
		number_of_removed_nodes = ThreadSafeCounter(0)
		while(not nodes_to_be_calculated == set()):
			print(len(nodes_to_be_calculated))
			new_nodes_to_be_calculated = set(nodes_to_be_calculated)
			
			calculation_processes = list()  # uncomment for multi-threading/ comment for single-threading
			
			nodes_that_can_be_calculated = []
			for node in nodes_to_be_calculated:
				if(self.can_node_be_calculated(node)):
					nodes_that_can_be_calculated.append(node)
					new_nodes_to_be_calculated.remove(node)
					new_nodes_to_be_calculated.update(self.nice_tree_decomposition.predecessors(node))
			p = multiprocessing.Pool(12)
			calculated_nodes = p.map(self.calculate_component_signature_of_node, nodes_that_can_be_calculated)
			p.close()
			p.join()
			for calculation, node in zip(calculated_nodes, nodes_that_can_be_calculated):
				self.component_signatures[node] = calculation
				
			nodes_to_be_calculated = new_nodes_to_be_calculated
			print("Done!")
			return True
		
		return component_signatures[self.root]


	def can_node_be_calculated(self, node):
		can_be_calculated = 0
		successors = list(self.nice_tree_decomposition.successors(node))
		for successor in successors:
			if(successor in self.component_signatures.keys()):
				can_be_calculated = can_be_calculated + 1
		
		return can_be_calculated == len(successors)

	
	def calculate_component_signature_of_node(self, node):
		if(node.node_type == ntd.Nice_Tree_Node.LEAF):
			successors = list(self.nice_tree_decomposition.successors(node))
			assert len(successors) == 0
			return self.find_component_signatures_of_leaf_nodes(node, node.bag)
		
		elif(node.node_type == ntd.Nice_Tree_Node.INTRODUCE):
			successors = list(self.nice_tree_decomposition.successors(node))
			assert len(successors) == 1
			child = successors[0]
			return self.calculate_component_signature_of_introduce_node(node, node.bag, child, child.bag, self.component_signatures[child])

		elif(node.node_type == ntd.Nice_Tree_Node.FORGET):
			successors = list(self.nice_tree_decomposition.successors(node))
			assert len(successors) == 1
			child = successors[0]
			return self.find_component_signatures_of_forget_node(node, child, self.component_signatures[child])

		else: 
			assert node.node_type == ntd.Nice_Tree_Node.JOIN
			successors = list(self.nice_tree_decomposition.successors(node))
			assert len(successors) == 2
			child_1 = successors[0]
			child_2 = successors[1]
			del_values_child = dict(self.component_signatures[child_1]).update(self.component_signatures[child_2])
			return find_component_signatures_of_join_nodes(node, node.bag, child_1, child_2, del_values_child)


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
			initial_partition.blocks.append(Block([]))

		partitions.add(initial_partition)

		for node in bag:
			new_partitions = set()
			
			for partition in partitions:
				# iterate over each block, using a for loop over an index because
				# we need the index for the blocks in the new partition we create
				for i in range(len(partition)):
					if len(partition[i]) < size:
						
						new_partition = partition.get_copy()
						
						new_partition[i].append(node)

						# check if identical partition already exists
						does_identical_partition_exist = False
						for existing_partition in new_partitions:
							is_a_block_different = False
							for block in existing_partition.blocks:
								if not block in new_partition:
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
			for block in partition:
				if block.node_list == [] :
					blocks_to_remove.append(block)
			
			for block in blocks_to_remove:
				partition.blocks.remove(block)

		return partitions

	# a function here is just a dictionary that maps its unique inputs values to output values
	def generate_all_functions_from_partition_to_range(self, partition, h):
		all_functions = set()
		all_functions.add(Function({})) # initial set of undefined functions used to generate the rest

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
		for (P, c) in all_states:
			potential_edges_to_remove = set()

			potential_edges_to_remove.update(self.edges_connecting_blocks_in_partition(bag, P))
			
			# for now I'm saving the del-values themselves because we'll probably need them
			# for the algorithm as suggested by the paper, the second value of the tuple is what we need
			if(len(potential_edges_to_remove) <= self.k):
				del_values[(leaf_node, (P, c))] = (potential_edges_to_remove, len(potential_edges_to_remove))
			else:
				# in this case, there should be no need to save the actual set of edges
				# so I'm leaving it as an empty set for performance reasons
				del_values[(leaf_node, (P, c))] = (set(), math.inf)
		
		return del_values

	# Algorithm 3
	def calculate_component_signature_of_introduce_node(self, introduce_node, bag, child, child_bag, del_values_child):
		del_Values = dict()
		all_States = self.generate_possible_component_states_of_bag(bag, self.h)

		# v is a set containing the node
		v = bag.symmetric_difference(child_bag)
		block_containing_v = set()

		# find block containing v
		for block in state[0]:
			if v.issubset(block):
				block_containing_v = block

		for state in all_States:
			introduce_inhereted_cStates = set()
			partition_without_block_containing_v = state[0].symmetric_difference(set(block_containing_v))
			refinements = self.generate_partitions_of_bag_of_size(block_containing_v.symmetric_difference(v), len(block_containing_v.symmetric_difference(v)))

			for refinement in refinements:
				partition_prime = partition_without_block_containing_v.union(refinement)
				all_Cs = self.algorithm3_function_generator(state[1], partition_without_block_containing_v, block_containing_v, refinement, self.h)

				for c_prime in all_Cs:
					introduce_inhereted_cStates.add((partition_prime, c_prime))

			minValue = math.inf
			minValue_edge_set = set()

			for state_prime in introduce_inhereted_cStates:
				# unsure whether state should be state_prime or not (paper says state)
				edge_set = edges_connecting_blocks_in_partition(self.graph.subgraph(bag).edges , state)
				value = del_values_child[(child, state_prime)] + len(edge_set)
				if value < minValue:
					minValue = value
					minValue_edge_set = edge_set

			if minValue <= k:
				del_Values[(introduce_node, state)] = (minValue_edge_set, minValue)
			else:
				del_Values[(introduce_node, state)] = (set(), math.inf)
		return del_Values

	# Algorithm 4
	def find_component_signatures_of_forget_node(self, node, child_node, del_values_child):
		return dict()

	# Algorithm 5
	def find_component_signatures_of_join_nodes(self, join_node, bag, child_1, child_2, del_values_child):
		del_values = dict()
		all_states = self.generate_possible_component_states_of_bag(bag, self.h)
		for (P, c) in all_states:
			sigma_t1_t2_join = set()
			partition_1 = P
			partition_2 = P
			all_function_pairs = self.get_all_function_pairs(P, c)

			for (c_1, c_2) in all_function_pairs:
				sigma_t1_t2_join.add(((partition_1, c_1), (partition_2, c_2)))
			
			minValue = math.inf
			minValueSet = set()
			for (sigma_1, sigma_2) in sigma_t1_t2_join:
				edges_connecting_blocks_in_partition = self.edges_connecting_blocks_in_partition(bag, P)
				value = (del_values_child[(child_1, sigma_1)][1] + del_values_child[(child_2, sigma_2)][1] 
						 - len(edges_connecting_blocks_in_partition))

				if(value < minValue):
					minValue = value
					minValueSet = edges_connecting_blocks_in_partition

			if(minValue <= self.k):
				del_values[join_node, (P, c)] = (minValueSet, minValue)
			else:
				del_values[join_node, (P, c)] = (set(), math.inf)

		return del_values
				

	def get_all_function_pairs(self, partition, c):
		all_function_pairs = set()
		all_function_pairs.add((Function(dict()), Function(dict()))) # initial set of undefined functions used to generate the rest
	
		# partially define function by setting the mapping for each block
		for block in partition:

			# for each possible value that a block could be mapped to
			new_all_function_pairs = []
			for c_1_codomain_value in range(1, self.h + 1):
				# go over each possible combination of c1 and c2 values
				for c_2_codomain_value in range(1, self.h + 1):
					# if that pair of values is legal
					if (c_1_codomain_value + c_2_codomain_value - len(block) == c[block]
						and c_1_codomain_value >= len(block) and c_2_codomain_value >= len(block)):
						# then for each function that already exists
						for function_pair in all_function_pairs:
							# create a new function with the only difference being
							# that the new block now has a mapping in that function

							new_function_pair = (Function(dict(function_pair[0].dictionary)), Function(dict(function_pair[1].dictionary)))
							new_function_pair[0][block] = c_1_codomain_value
							new_function_pair[1][block] = c_2_codomain_value
							new_all_function_pairs.append(new_function_pair)
			
			all_function_pairs = new_all_function_pairs

		return all_function_pairs

	# takes the edges of the induced subgraph and the current state
	# returns set of edges
	def	edges_connecting_blocks_in_partition(self, bag, partition):
		induced_subgraph_edges = self.graph.subgraph(list(bag)).edges
		result = set()
		for (u,w) in induced_subgraph_edges:
			block_containing_u = Block(set())
			block_containing_w = Block(set())
			for block in partition:
				if(u in block.node_list):
					block_containing_u = block
				if(w in block.node_list):
					block_containing_w = block
				
				if(block_containing_u.node_list != set() and block_containing_w.node_list != set()):
					break
				
			if(block_containing_u != block_containing_w):
				result.add((u,w))
		return result
	
	# a function here is just a dictionary that maps its unique inputs values to output values and fulfills the conditions from algorithm 3
	def algorithm3_function_generator(self, parent_function, partition_without_block_containing_v, block_containing_v, refinement, h):
		all_functions = set()
		basic_function = Function({})

		# each block from the original partition has to have the same value
		for block in partition_without_block_containing_v:
			basic_function.dictionary[block] = parent_function[block]
		
		# every function will be created with the basic_function as basis because every function has to fulfill the above condition
		all_functions.add(basic_function)

		c_of_block_containing_v = parent_function[block_containing_v]
		inserted_blocks = set()
		for block in refinement:

			for function in all_functions:

				sum_of_refinement_blocks_currently_assigned = 0
				for inserted_block in inserted_blocks:
					sum_of_refinement_blocks_currently_assigned += function[inserted_block]

				new_all_functions = set()

				# if block is the last block then its value is the remainder
				if len(refinement.blocks) - len(inserted_blocks) == 1:
					new_function = Function(dict(function.dictionary))
					new_function.dictionary[block] = c_of_block_containing_v - 1 - sum_of_refinement_blocks_currently_assigned
					new_all_functions.add(new_function)

				# - len(refinement.symmetric_difference(inserted_blocks)) is necassary because every c'(block)>=1
				else:
					for value in range(1, c_of_block_containing_v - 1 - sum_of_refinement_blocks_currently_assigned - len(set(refinement.blocks).symmetric_difference(inserted_blocks))):
						new_function = Function(dict(function.dictionary))
						new_function.dictionary[block] = value
						new_all_functions.add(new_function)

			all_functions = new_all_functions
			inserted_blocks.add(block)

		return all_functions
			



def test_function(node):
	return "TRUE"

# We need a wrapper class in order to have hashable lists for the set of Partitions
class Block:
	def __init__(self, node_list):
		self.node_list = node_list

	def __getitem__(self, key):
		return self.node_list[key]

	def __setitem__(self, key, value):
		self.node_list[key] = value

	def __len__(self):
		return len(self.node_list)

	def append(self, node):
		self.node_list.append(node)

	def __repr__(self):
		return "Block(%r)" % self.node_list

	def __eq__(self, other):
		if(isinstance(other, Block)):
			return self.node_list == other.node_list
		else:
			return False
	
	def __hash__(self):
		return hash(tuple(self.node_list))

class Partition:
	def __init__(self, blocks):
		self.blocks = blocks

	def __repr__(self):
		return "Partition(%r)" % self.blocks

	def get_copy(self):
		new_block_list = []
		for block in self.blocks:
			new_block_list.append(Block(list(block.node_list)))
		return Partition(new_block_list)

	def __getitem__(self, key):
		return self.blocks[key]
	
	def __setitem__(self, key, value):
		self.blocks[key] = value

	def __len__(self):
		return len(self.blocks)
	
	def __eq__(self, other):
		if(isinstance(other, Partition)):
			return self.blocks == other.blocks
		else:
			return False
	
	def __hash__(self):
		return hash(tuple(self.blocks))


class Function:
	def __init__(self, dictionary):
		self.dictionary = dictionary

	def __getitem__(self, key):
		return self.dictionary[key]

	def __setitem__(self, key, value):
		self.dictionary[key] = value

	def __eq__(self, other):
		if(isinstance(other, Function)):
			return self.dictionary == other.dictionary
		else:
			return False
	
	def __hash__(self):
		return hash(tuple(list(self.dictionary.keys()) + list(self.dictionary.values())))

	def __repr__(self):
		return "Function(%r)" % self.dictionary

class ThreadSafeCounter:
	def __init__(self, counter_value):
		self.counter = counter_value
		self.lock = threading.Lock()

	def increment(self):
		with self.lock:
			self.counter += 1

	def decrement(self):
		with self.lock:
			self.counter -= 1

class HashableDictionary(dict):
	def __key(self):
		return tuple((k,self[k]) for k in sorted(self))
	def __hash__(self):
		return hash(self.__key())
	def __eq__(self, other):
		return self.__key() == other.__key()

