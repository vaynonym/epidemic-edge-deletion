import copy
import math
import src.algorithms.nice_tree_decomposition as ntd
import networkx as nx
import multiprocessing
import threading
import sys
import traceback
import gc
import psutil
import os
from setproctitle import setproctitle 

from guppy import hpy

class Algorithm:
	
	def __init__(self, graph, nice_tree_decomposition, h, k):
		self.graph = graph
		self.h = h
		self.k = k
		self.nice_tree_decomposition = nice_tree_decomposition
		self.component_signatures = dict()
		setproctitle("Epidemic Edge Deletion")

	def execute(self):
		self.root = self.nice_tree_decomposition.root
		leafs = self.nice_tree_decomposition.find_leafs()

		process_count = min(4, len(leafs))

		result_queue = multiprocessing.Queue()

		processes = []
		work_queues = []
		calculated_nodes = multiprocessing.Value('i', 0)

		total_nodes = len(self.nice_tree_decomposition.graph.nodes)

		for i in range(process_count):
			wq = multiprocessing.Queue()
			p = multiprocessing.Process(target=worker, args=(i, self.graph, self.nice_tree_decomposition, self.h, self.k, wq, result_queue, calculated_nodes, total_nodes))
			p.start()
			processes.append(p)
			work_queues.append(wq)

		print("Created processes, starting work")

		for i in range(process_count):
			self.queue_node_and_preds(work_queues[i], leafs.pop(-1))

		while (calculated_nodes.value < total_nodes):
			# Read a single result
			(process_index, node, component_signature) = result_queue.get()
			
			print("Received a result from P%d" % process_index)

			#self.validate_signature(node, component_signature)

			# Store the component signature for later and track that the node is done
			self.component_signatures[node] = component_signature

			# Find all nodes which are now calculable because this one is done and queue them
			predecessors = self.nice_tree_decomposition.predecessors(node)
			for pred in predecessors:
				can_calculate_pred = True
				child_signatures = dict()
				children = list(self.nice_tree_decomposition.successors(pred))

				for child in children:
					if (not child in self.component_signatures.keys()):
						can_calculate_pred = False
					else:
						child_signatures.update(self.component_signatures[child])

				if (can_calculate_pred):
					self.queue_node_and_preds(work_queues[process_index], pred, children, child_signatures)
				elif (len(leafs) > 0):
					self.queue_node_and_preds(work_queues[process_index], leafs.pop(-1))

			# Print some intermediate status results
			print("Calculated %d nodes so far..." % calculated_nodes.value)

		for i in range(process_count):
			work_queues[i].put((None, None, None, False))

		for p in processes:
			p.join()

		return self.component_signatures[self.root]

	def execute_singlethreaded(self):
		self.root = self.nice_tree_decomposition.root
		leafs = self.nice_tree_decomposition.find_leafs()

		algWorker = AlgorithmWorker(self.graph, self.nice_tree_decomposition, self.h, self.k)

		calculated_nodes = 0
		total_nodes = len(self.nice_tree_decomposition.graph.nodes)

		calculatable_nodes = []
		calculatable_nodes.append((leafs.pop(-1), None, None))

		while (calculated_nodes < total_nodes):
			(node, children, child_component_signatures) = calculatable_nodes.pop(-1)

			print ("Starting %s node: %r" % (node.node_type, node))

			component_signature = algWorker.calculate_component_signature_of_node(node, children, child_component_signatures)

			#self.validate_signature(node, component_signature)

			# Store the component signature for later and track that the node is done
			self.component_signatures[node] = component_signature
			calculated_nodes += 1

			# Find all nodes which are now calculable because this one is done and queue them
			predecessors = self.nice_tree_decomposition.predecessors(node)
			for pred in predecessors:
				can_calculate_pred = True
				child_signatures = dict()
				children = list(self.nice_tree_decomposition.successors(pred))
				for child in children:
					if (not child in self.component_signatures.keys()):
						can_calculate_pred = False
					else:
						child_signatures.update(self.component_signatures[child])
				if (can_calculate_pred):
					self.queue_work_singlethreaded(calculatable_nodes, pred, children, child_signatures)
				elif (len(leafs) > 0):
					self.queue_work_singlethreaded(calculatable_nodes, leafs.pop(-1), None, None)

			# Print some intermediate status results
			print("Calculated %d nodes so far..." % calculated_nodes)

		return self.component_signatures[self.root]

	def queue_node_and_preds(self, work_queue, node, initial_children = None, initial_signature = None):
		if (initial_children != None):
			for child in initial_children:
				del self.component_signatures[child]

		predecessors = list(self.nice_tree_decomposition.predecessors(node))
		if (predecessors == None or len(predecessors) == 0):
			work_queue.put((node, initial_children, initial_signature, False))
			return

		pred = predecessors[0]

		if pred.node_type != ntd.Nice_Tree_Node.JOIN:
			work_queue.put((node, initial_children, initial_signature, True))
			self.queue_node_and_preds(work_queue, pred)
		else:
			work_queue.put((node, initial_children, initial_signature, False))


	def queue_work_singlethreaded(self, calculatable_nodes, node, children, child_signatures):
		calculatable_nodes.append((node, children, child_signatures))

		if (children == None):
			 return

		for child in children:
			del self.component_signatures[child]

	def validate_signature(self, node, signature):
		# Compute G[V_t]
		v_t = self.get_v_t(node)

		for (edges, val) in signature.values():
			if (val == math.inf and edges == set()):
				continue

			if (val != len(edges)):
				print("Found a non-matching pair: %f, %s" % (val, edges))
				print("Node was: %r" % node)

			# Remove edges in signature from G[V_t]
			subgraph = nx.Graph(self.graph.subgraph(v_t))
			subgraph.remove_edges_from(edges)
			# Check if largest component is small enough
			max_component = max([len(c) for c in nx.connected_components(subgraph)])

			if (max_component > self.h):
				print("\n\nFound a pair that leaves a too-large component:")
				self.print_node_info(node, v_t, (val, edges))

			# Is the greatest component in G[node.bag] <= self.h?
			#components = nx.connected_components(self.graph.subgraph(node.bag))
			#max_component = max([len(c) for c in components])
			#is_bag_with_all_edges_valid = max_component <= self.h

			#if (edges == set() and not is_bag_with_all_edges_valid):
			#	print("Found an empty set that can't be empty: %f, %s" % (val, edges))
			#	print("Node was: %r" % node)

	def print_node_info(self, node, v_t, sig):
		(val, edges) = sig
		print("error signature: %f, %s" % (val, edges))
		print("from node: %r" % node)
		print("with v_t: %r" % v_t)
		print("children:")
		for child in self.nice_tree_decomposition.successors(node):
			self.print_node_info_complete(child)

	def print_node_info_complete(self, node, prefix = ""):
		print("%snode: %r" % (prefix, node))
		print("%scomp_signature: %r" % (prefix, self.component_signatures[node]))
		for child in self.nice_tree_decomposition.successors(node):
			self.print_node_info_complete(child, prefix + "\t")

	def get_v_t(self, node):
		v_t = set()
		v_t.update(node.bag)
		for s in self.nice_tree_decomposition.successors(node):
			v_t.update(self.get_v_t(s))
		return v_t

	def can_node_be_calculated(self, node):
		can_be_calculated = 0
		successors = self.nice_tree_decomposition.successors(node)
		for successor in successors:
			if (not successor in self.component_signatures.keys()):
				return False
		
		return True

def worker(process_index, graph, nice_tree_decomposition, h, k, work_queue, result_queue, calculated_nodes, total_nodes):
	setproctitle("Epidemic Edge Deletion Worker %d" % process_index)

	#this_process = psutil.Process(os.getpid())
	#hp = hpy()
		
	alg = AlgorithmWorker(graph, nice_tree_decomposition, h, k)

	#hp.setrelheap()

	(node, _, _, has_more_in_branch) = work_queue.get()
	last_comp_signature = None
	last_node = None
	while (node != None):
		try:
			print ("[P%d] Worker starting %s node (%d/%d): %r" % (process_index, node.node_type, calculated_nodes.value + 1, total_nodes, node))

			if (node.node_type == ntd.Nice_Tree_Node.JOIN):
				last_comp_signature = alg.calculate_component_signature_of_node(node, join_children, join_signatures)
			else:
				last_comp_signature = alg.calculate_component_signature_of_node(node, [last_node], last_comp_signature)

			last_node = node

			with calculated_nodes.get_lock():
				calculated_nodes.value += 1

			if (not has_more_in_branch):
				result_queue.put((process_index, node, last_comp_signature))

			(node, join_children, join_signatures, has_more_in_branch) = work_queue.get()

			#result_queue.put((process_index, node, component_signature))
			#print ("[P%d] Worker finished %s node: %r" % (process_index, node.node_type, node))

			#node = None
			#children = None
			#child_component_signatures = None
			#component_signature = None

			#print("[P%d] Memory usage before GC: %d" % (process_index, this_process.memory_info().rss))
			#gc.collect()
			#print("[P%d] Memory usage after GC: %d" % (process_index, this_process.memory_info().rss))
			#print("[P%d] Heap: %s" % (process_index, hp.heap()))
		except KeyError as e:
			print("\n====================\nEncountered error: %s" % e)
			print("Traceback: %s" % traceback.format_exc())
			print("node is %r" % node)
			print("children is %r" % (join_children if node.node_type == ntd.Nice_Tree_Node.JOIN else [last_node]))
			print("child_component_signatures is %r" % (join_signatures if node.node_type == ntd.Nice_Tree_Node.JOIN else last_comp_signature))
		#except KeyboardInterrupt as i:
		#	print("[P%d] Received an interrupt, cancelling. Heap:\n%s" % (process_index, hp.heap()))
		#	raise i

class AlgorithmWorker:
	def __init__(self, graph, nice_tree_decomposition, h, k):
		self.graph = graph
		self.h = h
		self.k = k
		self.nice_tree_decomposition = nice_tree_decomposition
	
	def calculate_component_signature_of_node(self, node, children, child_component_signatures):
		if(node.node_type == ntd.Nice_Tree_Node.LEAF):
			return self.find_component_signatures_of_leaf_nodes(node, node.bag)
		
		elif(node.node_type == ntd.Nice_Tree_Node.INTRODUCE):
			child = children[0]
			return self.calculate_component_signature_of_introduce_node(node, node.bag, child, child.bag, child_component_signatures)

		elif(node.node_type == ntd.Nice_Tree_Node.FORGET):
			child = children[0]
			return self.find_component_signatures_of_forget_node(node, child, child_component_signatures)

		else: 
			assert node.node_type == ntd.Nice_Tree_Node.JOIN
			child_1 = children[0]
			child_2 = children[1]
			return self.find_component_signatures_of_join_nodes(node, node.bag, child_1, child_2, child_component_signatures)


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
			initial_partition.append(Block([]))

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
						new_partition.sort()

						# check if identical partition already exists
						does_identical_partition_exist = False
						for existing_partition in new_partitions:
							is_a_block_different = False
							for block in existing_partition:
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
				if len(block) == 0 :
					blocks_to_remove.append(block)
			
			for block in blocks_to_remove:
				partition.remove(block)

		return partitions

	# a function here is just a dictionary that maps its unique inputs values to output values
	# def generate_all_functions_from_partition_to_range(self, partition, h):
	# 	all_functions = set()
	# 	all_functions.add(Function({})) # initial set of undefined functions used to generate the rest

	# 	# partially define function by setting the mapping for each block
	# 	for block in partition:

	# 		# for each possible value that a block could be mapped to
	# 		new_all_functions = []
	# 		for codomain_value in range(len(block), h+1):
	# 			# then for each function that already exists
	# 			for function in all_functions:
	# 				# create a new function with the only difference being
	# 				# that the new block now has a mapping in that function

	# 				new_function = Function(dict(function.dictionary))
	# 				new_function[block] = codomain_value
	# 				if(codomain_value == h):
	# 					yield new_function
	# 				else {
	# 					new_all_functions.append(new_function)
	# 				}
			
	# 		all_functions = new_all_functions

	# 	return all_functions

	def generate_all_functions_from_partition_to_range(self, partition, h):
		min_Values_Of_Blocks = []
		for block in partition:
			min_Values_Of_Blocks.append(len(block))
		
		current_Values_Of_Blocks = list(min_Values_Of_Blocks)

		last_Values_Of_Blocks = [h] * len(partition)

		while(not current_Values_Of_Blocks == last_Values_Of_Blocks):
			function = dict()
			for i in range(len(partition)):
				function[partition[i]] = current_Values_Of_Blocks[i]
			yield Function(function)

			for i in range(len(current_Values_Of_Blocks)):
				if(not current_Values_Of_Blocks[i] == h):
					current_Values_Of_Blocks[i] += 1
					for j in range(i):
						current_Values_Of_Blocks[j] = min_Values_Of_Blocks[j]
					break
		
		function = dict()
		for i in range(len(partition)):
			function[partition[i]] = current_Values_Of_Blocks[i]
		yield Function(function)
		


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
		# v is not a set anymore
		v = list(v)[0]
		block_containing_v = set()

		for state in all_States:
			introduce_inhereted_cStates = set()

			# find block containing v
			for block in state[0]:
				if v in block:
					block_containing_v = block

			partition_without_block_containing_v = Partition(list(state[0].symmetric_difference(Partition([block_containing_v]))))

			block_without_v = Block(block_containing_v.symmetric_difference(Block([v])))
			refinements = self.generate_partitions_of_bag_of_size(block_without_v, len(block_without_v))

			for refinement in refinements:
				partition_prime = partition_without_block_containing_v.union(refinement)
				all_Cs = self.algorithm3_function_generator(state[1], partition_without_block_containing_v, block_containing_v, refinement, self.h)

				for c_prime in all_Cs:
					introduce_inhereted_cStates.add((partition_prime, c_prime))

			minValue = math.inf
			minValue_edge_set = set()

			for state_prime in introduce_inhereted_cStates:
				# unsure whether state should be state_prime or not (paper says state)
				edge_set = self.edges_connecting_node_with_other_block_in_partition(v, bag, state[0])
				value = (del_values_child[(child, state_prime)])[1] + len(edge_set)
				if value < minValue:
					minValue = value
					minValue_edge_set = edge_set.union((del_values_child[(child, state_prime)])[0])
					#if (len(minValue_edge_set) != minValue):
					#	print("\n\nOH NO: minValue: %f, edge set: %r" % (minValue, minValue_edge_set))
					#	print("child_set: %r" % (del_values_child[(child, state_prime)])[0])
					#	print("edge_set: %r\n\n" % edge_set)

			if minValue <= self.k:
				del_Values[(introduce_node, state)] = (minValue_edge_set, minValue)
			else:
				del_Values[(introduce_node, state)] = (set(), math.inf)
		return del_Values

	# Algorithm 5
	def find_component_signatures_of_join_nodes(self, join_node, bag, child_1, child_2, del_values_child):
		del_values = dict()
		all_states = self.generate_possible_component_states_of_bag(bag, self.h)
		for (P, c) in all_states:
			sigma_t1_t2_join = set()
			partition_1 = P
			partition_2 = P
			all_function_pairs = self.get_all_function_pairs(P, c)
			edges_connecting_blocks_in_partition = self.edges_connecting_blocks_in_partition(bag, P)

			for (c_1, c_2) in all_function_pairs:
				sigma_t1_t2_join.add(((partition_1, c_1), (partition_2, c_2)))
			
			minValue = math.inf
			minValueSet = set()
			for (sigma_1, sigma_2) in sigma_t1_t2_join:
				tuple_child_1 = del_values_child[(child_1, sigma_1)]
				tuple_child_2 = del_values_child[(child_2, sigma_2)]

				value = (tuple_child_1[1] + tuple_child_2[1] 
						 - len(edges_connecting_blocks_in_partition))

				if (value == -1):
					print("\n___________________________________\n")
					print("Found a -1 value!")
					print("Node: %r" % join_node)
					print("Child1: %r" % child_1)
					print("tuple1:{}".format(tuple_child_1))
					print("Child2: %r" % child_2)
					print("tuple2: {}".format(tuple_child_2))
					print("edges_connecting: %r" % edges_connecting_blocks_in_partition)

				if(value < minValue):
					minValue = value
					# minValueSet = edges_connecting_blocks_in_partition
					minValueSet = tuple_child_1[0].union(tuple_child_2[0])#.difference(edges_connecting_blocks_in_partition)
					#edges_connecting_blocks_in_partition

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
			block_containing_u = None
			block_containing_w = None
			for block in partition:
				if(u in block):
					block_containing_u = block
				if(w in block):
					block_containing_w = block
				
				if(block_containing_u != None and block_containing_w != None):
					break
				
			if(block_containing_u != block_containing_w):
				result.add((u,w))
		return result

	def edges_connecting_node_with_other_block_in_partition(self, node, bag, partition):
		induced_subgraph_edges = self.graph.subgraph(list(bag)).edges
		result = set()

		block_containing_node = None
		for block in partition:
			if (node in block):
				block_containing_node = block
				break

		for (u,w) in induced_subgraph_edges:
			if (u == node):
				to_search = w
			elif (w == node):
				to_search = u
			else:
				continue

			block_containing_to_search = None
			for block in partition:
				if(to_search in block):
					block_containing_to_search = block
					break

			if (block_containing_node != block_containing_to_search):
				result.add((u,w))

		return result
	
	# a function here is just a dictionary that maps its unique inputs values to output values and fulfills the conditions from algorithm 3
	def algorithm3_function_generator(self, parent_function, partition_without_block_containing_v, block_containing_v, refinement, h):
		all_functions = set()
		basic_function = Function({})

		# each block from the original partition has to have the same value
		for block in partition_without_block_containing_v:
			basic_function[block] = parent_function[block]
		
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
				if len(refinement) - len(inserted_blocks) == 1:
					new_function = Function(dict(function.dictionary))
					new_function[block] = c_of_block_containing_v - 1 - sum_of_refinement_blocks_currently_assigned
					new_all_functions.add(new_function)

				# - len(refinement.symmetric_difference(inserted_blocks)) is necassary because every c'(block)>=1
				else:
					for value in range(len(block), c_of_block_containing_v - 1 - sum_of_refinement_blocks_currently_assigned - len(set(refinement.blocks).symmetric_difference(inserted_blocks))):
						new_function = Function(dict(function.dictionary))
						new_function[block] = value
						new_all_functions.add(new_function)

			all_functions = new_all_functions
			inserted_blocks.add(block)

		return all_functions

	# Algorithm 4
	def find_component_signatures_of_forget_node(self, node, child_node, del_values_child):
		del_values = dict()
		all_states = self.generate_possible_component_states_of_bag(node.bag, self.h)

		child_extra_nodes = child_node.bag - node.bag
		if (len(child_extra_nodes) != 1):
			 raise ValueError

		(v,) = child_extra_nodes

		for state in all_states:
			P = state[0]
			c = state[1]

			child_forget_inherited_states = list()
			all_child_partitions = self.get_all_extended_partitions(P, v)

			for child_partition in all_child_partitions:
				all_child_functions = list()
				child_c = Function(dict())
				vSingleton = False

				for block in child_partition:
					block_without_v = Block(list(block.node_list))
					if (v in block_without_v):
						block_without_v.remove(v)

					if len(block_without_v) == 0:
						vSingleton = True
					else:
						# Algorithm from paper just reads 'child_c[block] = c[block_without_v]' here, which can
						# result in invalid child_c functions (which the paper just ignores).
						# We check all functions for validity below, before proceeding with the minimum search.
						child_c[block] = c[block_without_v]
				
				if not vSingleton:
					all_child_functions.append(child_c)
				else:
					for i in range(1, self.h + 1):
						c_copy = Function(dict(child_c.dictionary))
						c_copy[Block([v])] = i
						all_child_functions.append(c_copy)

				# This doesn't quite match the pseudo-code, but it makes more sense and matches the
				# reference implementation.
				for c_i in all_child_functions:
					if (self.is_valid_function(child_partition, c_i)):
						child_forget_inherited_states.append((child_partition, c_i))

			min_value = math.inf
			min_set = set()
			for sigma in child_forget_inherited_states:
				(val_set, val) = del_values_child[(child_node, sigma)]
				if val < min_value:
					min_value = val
					min_set = val_set
			if min_value <= self.k:
				del_values[(node, state)] = (min_set, min_value)
			else:
				del_values[(node, state)] = (set(), math.inf)
		return del_values

	def get_all_extended_partitions(self, partition, new_node):
		extended_partitions = list()
		for i in range(len(partition)):
			p = partition.get_copy()
			p[i].append(new_node)
			p.sort()
			extended_partitions.append(p)
		p = partition.get_copy()
		p.append(Block([new_node]))
		p.sort()
		extended_partitions.append(p)
		return extended_partitions

	def is_valid_function(self, partition, function):
		for block in partition:
			if (function[block] < len(block)):
				return False
		return True

# We need a wrapper class in order to have hashable lists for the set of Partitions
class Block:
	def __init__(self, node_list):
		self.node_list = sorted(node_list)
		self.hash = None
		self.calculate_hash()
		
	def __getitem__(self, key):
		return self.node_list[key]

	def __setitem__(self, key, value):
		self.node_list[key] = value
		self.node_list = sorted(self.node_list)
		self.calculate_hash()

	def __len__(self):
		return len(self.node_list)

	def append(self, node):
		self.node_list.append(node)
		self.node_list = sorted(self.node_list)
		self.calculate_hash()

	def remove(self, node):
		self.node_list.remove(node)
		self.node_list = sorted(self.node_list)
		self.calculate_hash()

	def __repr__(self):
		return "Block(%r)" % self.node_list

	def __eq__(self, other):
		if(isinstance(other, Block)):
			return self.node_list == other.node_list
		else:
			return False
	
	def __lt__(self, other):
		if not isinstance(other, Block):
			return NotImplemented
		if (len(self.node_list) == 0):
			return True
		if (len(other.node_list) == 0):
			return False
		return self.node_list[0] < other.node_list[0]

	def __hash__(self):
		return self.hash
	
	def calculate_hash(self):
		self.hash = hash(tuple(self.node_list))

	def symmetric_difference(self, block):
		return set(self.node_list).symmetric_difference(set(block.node_list))

class Partition:
	def __init__(self, blocks):
		self.blocks = sorted(blocks)
		self.hash = None
		self.calculate_hash()

	def __repr__(self):
		return "Partition(%r)" % self.blocks

	def get_copy(self):
		new_block_list = []
		for block in self.blocks:
			new_block_list.append(Block(list(block.node_list)))
		return Partition(new_block_list)

	# It's not great that we ever need to do this manually, would be good to avoid
	# the calls to this when generating partitions.
	def sort(self):
		self.blocks = sorted(self.blocks)
		self.calculate_hash()

	def __getitem__(self, key):
		return self.blocks[key]
	
	def __setitem__(self, key, value):
		self.blocks[key] = value
		self.sort()

	def remove(self, value):
		self.blocks.remove(value)
		self.sort()

	def append(self, value):
		self.blocks.append(value)
		self.sort()

	def __len__(self):
		return len(self.blocks)

	def __eq__(self, other):
		if(isinstance(other, Partition)):
			return self.blocks == other.blocks
		else:
			return False
	
	def __hash__(self):
		return self.hash

	def calculate_hash(self):
		self.hash = hash(tuple(self.blocks))

	def symmetric_difference(self, partition):
		return set(self.blocks).symmetric_difference(set(partition.blocks))

	def union(self, partition):
		return Partition(list(set(self.blocks).union(set(partition.blocks))))


class Function:
	def __init__(self, dictionary):
		self.dictionary = dictionary
		self.hash = None
		self.calculate_hash()

	def __getitem__(self, key):
		return self.dictionary[key]

	def __setitem__(self, key, value):
		self.dictionary[key] = value
		self.calculate_hash()

	def __eq__(self, other):
		if(isinstance(other, Function)):
			# TODO: Can this be done more efficiently?
			for key, value in self.dictionary.items():
				if not key in other.dictionary:
					return False
				if value != other.dictionary[key]:
					return False
			for key, value in other.dictionary.items():
				if not key in self.dictionary:
					return False
				if value != self.dictionary[key]:
					return False
			return True
		else:
			return False

	def __hash__(self):
		return self.hash

	def calculate_hash(self):
		#self.hash = hash(tuple(sorted(self.dictionary.keys()) + sorted(self.dictionary.values())))
		self.hash = hash(tuple(sorted(self.dictionary.items(), key = lambda p: p[0])))

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