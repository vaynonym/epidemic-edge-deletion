import networkx as nx
import numpy as np
import random
from networkx.algorithms import approximation
import copy
import matplotlib.pyplot as mpl
#import src.algorithms.nice_tree_decomposition as ntd
import src.algorithms.nice_tree_decomposition as ntd

class Tree_Decomposer:
	
	def __init__(self, graph):
    	# generates random graph on 30 nodes and a heuristic creates a TD
		# TODO: replace with self.graph = graph and call function to compute TD
		self.TD = nx.algorithms.approximation.treewidth.treewidth_min_degree(nx.fast_gnp_random_graph(20, 1/2, None, False))
		self.graph = self.TD[1]
		self.niceTD = nx.DiGraph()
	
	def make_nice_tree_decomposition(self):
		# don't change order of these three function calls otherwise make_nice_Tree_nodes won't work proberly
		self.choose_root()
		self.make_directed()
		self.make_nice_tree_nodes()

		queue = [self.graph_root]
		while len(queue)>0:
			node = queue.pop(0)
			# TODO: special case bag of child equals bag of parent ?
			
			# if the node is a leaf then nothing has to be done
			if len(list(self.graph.successors(node))) == 0:
				continue
			# these three function calls will construct the niceTD structur from a parent to each children
			self.create_introduce_nodes(node)
			# if parent has only one child insert missing vertices (forget nodes between child and parent)
			if len(list(self.graph.successors(node))) == 1:
				# if forget node property already fullfilled no need to call create_forget_nodes
				if not len(node.bag) + 1 == len(list(self.graph.successors(node))[0].bag):
					self.create_forget_nodes(node)
			# technically this if statement should not be neccessary i just put it there in desperation
			if len(list(self.graph.successors(node))) >= 2:
				self.create_join_node(node)

			queue.extend(list(self.graph.successors(node)))
		return self.niceTD
	
	# chooses root of niceTD and adds it to niceTD
	def choose_root(self):
    	# for now chosen randomly
		# list(graph) returns a list of the nodes (random.choice needs iterable object)
		self.graph_root = random.choice(list(self.graph.nodes))

	# when a node in the TD contains a vertex in its bag that none of its neighbours' bags contain then create introduce nodes until this is not true anymore
	def create_introduce_nodes(self, node):
		last_node=node
		union_of_children = set()
		for child in set(self.graph.successors(node)):
			union_of_children = union_of_children.union(child.bag)
		
		node_without_children = node.bag
		for elem in node.bag.intersection(union_of_children):
			node_without_children.discard(elem)

		new_parent = node
		if not set() == node_without_children:
			node_without_children = list(node_without_children)
			while len(node_without_children)>0:
				vertex = node_without_children.pop(0)
				temp = last_node.bag
				temp.remove(vertex)
				last_node = ntd.Nice_Tree_Node(temp)
				self.niceTD.add_edge(new_parent, last_node)
				new_parent=last_node

		# add new parent node to self.graph otherwise queue in make_nice_tree_decomposition won't work properly
		if last_node != node:
			if len(list(self.graph.successors(node))) >= 1:
				for child in list(self.graph.successors(node)):
					self.graph.add_edge(new_parent, child)
					self.graph.remove_edge(node, child)
				self.graph.add_edge(node, new_parent)
		return node

	# TODO: cut out some lines because this function will only be called if node hast exactly one child -> superfluous code
	# when a node in the TD does not contain a vertex that is in the intersection of the childrens' bags then create forget nodes until this is not true anymore
	def create_forget_nodes(self, node):
		last_node=node
		intersection_of_children = random.choice(list(self.graph.successors(node))).bag
		for child in set(self.graph.successors(node)):
			intersection_of_children = intersection_of_children.intersection(child.bag)
		if not set() == intersection_of_children:
			for vertex in intersection_of_children:
				temp = last_node.bag
				temp.add(vertex)
				last_node = ntd.Nice_Tree_Node(temp)
				self.niceTD.add_edge(node, last_node)
				node=last_node
		return node

	# when a node in the TD is subset of the union of the children then we need to create a join node and build to subtrees
	def create_join_node(self, node):
		# TODO: this does not work probably because deepcopires still have the same hash
		left_node = ntd.Nice_Tree_Node(node.bag)
		right_node = ntd.Nice_Tree_Node(node.bag)
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
		self.graph.add_edge(node, left_node)
		self.graph.add_edge(node, right_node)

		for left_child in best_partition[0]:
			self.graph.add_edge(left_node, left_child)
			self.graph.remove_edge(node, left_child)

		for right_child in best_partition[1]:
			self.graph.add_edge(right_node, right_child)
			self.graph.remove_edge(node, right_child)

		return [left_node, right_node]

	# returns list of partitions with 2 blocks as [partition] where partition is a list of 2 sets
	def generate_partitions_with_2_blocks(self, arg_set , nonempty=True):
		if len(arg_set)==0:
			return
		first = list(arg_set)[0]
		arg_set.discard(first)
		list_of_partitions = [[[first],[]]]
		temp = [[[first],[]]]

		for x in arg_set:
			temp=[]
			for partial_partition in list_of_partitions:
				pp_left = []
				pp_right = []
				for block in partial_partition:
					pp_right.append(list(block))
					pp_left.append(list(block))
				pp_left[0].append(x)
				pp_right[1].append(x)
				temp.append(pp_left)
				temp.append(pp_right)

			list_of_partitions = temp
		# removes partition where one block is empty if nonempty is true
		if nonempty:
			for partition in list_of_partitions:
				if partition[0]==[] or partition[1]==[]:
					list_of_partitions.remove(partition)
		return list_of_partitions

	# takes a partition and evalutes the number of introduce
	def evaluate_partition(self, node, partition):
		# computes the number of introduce nodes one can save using this partition
		union_of_children_left = set()
		for x in partition[0]:
    			union_of_children_left.union(x.bag)
		node_diff_left = node.bag.difference(node.bag.intersection(union_of_children_left))

		union_of_children_right = set()
		for x in partition[1]:
    			union_of_children_right.union(x.bag)
		node_diff_right = node.bag.difference(node.bag.intersection(union_of_children_right))
		# technically, this value has to be decreased by one
		introduce_nodes_saved = len(node_diff_left) + len(node_diff_right)

		# i am not 100% sure if this quantity is actually relevant
		intersection_of_children_left = random.choice(random.choice(partition)).bag
		for x in partition[0]:
    			intersection_of_children_left = intersection_of_children_left.intersection(x.bag)

		intersection_of_children_right = random.choice(random.choice(partition)).bag
		for x in partition[1]:
    			intersection_of_children_right = intersection_of_children_right.intersection(x.bag)

		return (introduce_nodes_saved, len(intersection_of_children_left)+len(intersection_of_children_right))

	# transforms an undirected TD to a directed TD where only parents have an edge to its child(ren)
	def make_directed(self):
		self.graph = self.graph.to_directed()
		queue = [self.graph_root]
		while len(queue)>0:
			parent = queue[0]
			for child in list(self.graph.successors(parent)):
					self.graph.remove_edge(child, parent)
			queue.pop(0)
			queue = queue + list(self.graph.successors(parent))

	# transform every node of self.graph into a Nice_Tree_Node and updates self.graph_root
	def make_nice_tree_nodes(self):
		graph = nx.DiGraph()
		node_pair_dict = {}
		for node in list(self.graph.nodes):
			gnode = ntd.Nice_Tree_Node(set(node))
			node_pair_dict[node] = gnode
		for node in node_pair_dict:
			for edge in list(self.graph.edges(node)):
				graph.add_edge(node_pair_dict[edge[0]],node_pair_dict[edge[1]])
		self.graph = graph
		self.graph_root = node_pair_dict[self.graph_root]
		self.niceTD.add_node(self.graph_root)


treed = Tree_Decomposer(nx.Graph())
a=ntd.Nice_Tree_Node({1,2,3})
b=ntd.Nice_Tree_Node({1,4,3})
c=ntd.Nice_Tree_Node({6,2,3})
d=ntd.Nice_Tree_Node({1,2,7})

treed.make_nice_tree_decomposition()
nx.draw(treed.niceTD)
mpl.show()
