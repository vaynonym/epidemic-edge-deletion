import networkx as nx
import numpy as np
import random
from networkx.algorithms import approximation
import copy

class Tree_Decomposer:
	
	def __init__(self, graph):
    	# generates random graph on 30 nodes and a heuristic creates a TD
		# TODO: replace with self.graph = graph and call function to compute TD
		self.TD = nx.algorithms.approximation.treewidth.treewidth_min_degree(nx.fast_gnp_random_graph(30, 1/2, None, False))
		self.graph = self.TD[1]
		self.niceTD = nx.DiGraph()
	
	def make_nice_tree_decomposition(self):
		self.choose_root()
		self.make_directed()
		queue = [self.graph_root]
		nodes_seen = []
		for node in queue:
			queue = queue + [self.graph.successors(node)]
			nodess_seen = nodes_seen + [node]
			queue = [x for x in queue if x not in nodes_seen]
			# TODO: special case bag of child equals bag of parent ?
			# if the node is a leaf then nothing has to be done
			if len(list(self.graph.successors(node))) == 1:
				continue
			# these three function calls will construct the niceTD structur from the node to each of the children
			self.create_introduce_nodes(node)
			if self.graph.successors(node) == 1:
				self.create_forget_nodes(node)
			self.create_join_node(node)
		return self.niceTD
	
	# chooses root of niceTD and adds it to niceTD
	def choose_root(self):
    	# for now chosen randomly
		# list(graph) returns a list of the nodes (random.choice needs iterable object)
		self.graph_root = random.choice(list(self.graph))
		self.niceTD.add_node(self.graph_root)

	# when a node in the TD contains a vertex in its bag that none of its neighbours' bags contain then create introduce nodes until this is not true anymore
	def create_introduce_nodes(self, node):
		last_node=node
		union_of_children = frozenset()
		for x in set(self.graph.successors(node)):
    			union_of_children.union(x)
		node_without_children = set(node)
		node_without_children.discard(node.intersection(union_of_children))
		# frozenset() == set() returns true
		if not frozenset() == node_without_children:
			for vertex in node_without_children:
				temp = set(last_node)
				temp.discard(vertex)
				last_node=frozenset(temp)
				#last_node.discard(vertex)
				self.niceTD.add_edge(node, last_node)
				node=last_node
		return node

	# TODO: cut out some lines because this function will only be called if node hast exactly one child -> superfluous code
	# when a node in the TD does not contain a vertex that is in the intersection of the childrens' bags then create forget nodes until this is not true anymore
	def create_forget_nodes(self, node):
		last_node=node
		intersection_of_children = random.choice(list(self.graph.successors(node)))
		for x in set(self.graph.successors(node)):
			intersection_of_children.intersection(x)
		if not frozenset() == intersection_of_children:
			for vertex in intersection_of_children:
				temp=set(last_node)
				temp.add(vertex)
				last_node=frozenset(temp)
				# das geht nicht, weil add(...) das set nicht returned
				#last_node=frozenset(set(last_node).add(vertex))
				self.niceTD.add_edge(node, last_node)
				node=last_node
		return node

	# when a node in the TD is subset of the union of the children then we need to create a join node and build to subtrees
	def create_join_node(self, node):
		left_node = copy.deepcopy(node)
		right_node = copy.deepcopy(node)
		self.niceTD.add_edge(node, left_node)
		self.niceTD.add_edge(node, right_node)

    	# computes how to separate the children
		best_partition = []
		best_partition_value = (0,0)
		list_of_partitions = self.generate_partitions_with_2_blocks(set(self.graph.successors(node)),2)
		for partition in list_of_partitions:
			new_partition_value = self.evaluate_partition(node, partition)

			if best_partition_value[0] == new_partition_value[0]:
				if best_partition_value[0] <= new_partition_value[0]:
					best_partition_value = new_partition_value
					best_partition = partition
			if best_partition_value[0] <= new_partition_value[0]:
				best_partition_value = new_partition_value
				best_partition = partition
		
		# assigns children to left and right node
		for left_child in best_partition[0]:
			self.niceTD.add_edge(left_node, left_child)

		for right_child in best_partition[1]:
			self.niceTD.add_edge(right_node, right_child)

		return [left_node, right_node]

	# returns list of partitions with 2 blocks as [partition] where partition is a list of 2 sets
	def generate_partitions_with_2_blocks(self, arg_set , nonempty=True):
		first = list(arg_set)[0]
		arg_set.discard(first)
		list_of_partitions = [[{first},set()]]
		temp = [[{first},set()]]
		for x in arg_set:
			for partiel_partition in list_of_partitions:
				pp_left = copy.deepcopy(partiel_partition)
				pp_right = copy.deepcopy(partiel_partition)
				pp_left[0].add(x)
				pp_right[1].add(x)
				temp.remove(partiel_partition)
				temp.append(pp_left)
				temp.append(pp_right)
			list_of_partitions = copy.deepcopy(temp)
		# removes partition where one block is empty if nonempty is true
		if nonempty:
			for partition in list_of_partitions:
				if partition[0]==set() or partition[1]==set():
					list_of_partitions.remove(partition)
		return list_of_partitions

	# takes a partition and evalutes the number of introduce
	def evaluate_partition(self, node, partition):
		# computes the number of introduce nodes one can save using this partition
		union_of_children_left = frozenset()
		for x in partition[0]:
    			union_of_children_left.union(x)
		node_diff_left = node.difference(node.intersection(union_of_children_left))

		union_of_children_right = frozenset()
		for x in partition[1]:
    			union_of_children_right.union(x)
		node_diff_right = node.difference(node.intersection(union_of_children_right))
		# technically, this value has to be decreased by one
		introduce_nodes_saved = len(node_diff_left) + len(node_diff_right)

		# i am not 100% sure if this quantity is actually relevant
		intersection_of_children_left = random.choice(list(partition))
		for x in partition[0]:
    			intersection_of_children_left = intersection_of_children_left.intersection(x)

		intersection_of_children_right = random.choice(list(self.graph.successors(node)))
		for x in partition[1]:
    			intersection_of_children_right = intersection_of_children_right.intersection(x)

		return (introduce_nodes_saved, len(intersection_of_children_left)+len(intersection_of_children_right))

	# transforms an undirected TD to a directed TD where only parents have an edge to its child(ren)
	def make_directed(self):
		self.graph = self.graph.to_directed()
		queue = [self.graph_root]
		for parent in queue:
			for child in list(self.graph.predecessors(parent)):
					self.graph.remove_edge(child, parent)
			queue = queue + [self.graph.successors(parent)]
