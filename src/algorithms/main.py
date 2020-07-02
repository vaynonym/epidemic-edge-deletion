import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import src.preprocessing.preprocessor as prepro
import src.algorithms.tree_decomposition as td
import src.algorithms.algorithm as algo
import src.algorithms.nice_tree_decomposition as ntd
import math
import sys
from argparse import ArgumentParser

def print_graph(graph, filename, planar):
	plt.figure(num=None, figsize=(35, 35), dpi=128)
	if (planar):
		nx.draw_planar(graph, node_size=25)
	else:
		nx.draw(graph, node_size=25)

	plt.savefig(filename)

def main(h, k, state_filter):

	print("  __________________________")
	print(" /		            \\")
	print("|     Running Program...     |")
	print(" \\__________________________/")
	print()

	print("  __________________________")
	print(" /		            \\")
	print("|        Loading Data...     |")
	print(" \\__________________________/")
	print()

	preprocessor = prepro.Preprocessor()
	graph, identifier_to_district_dictionary, position_dictionary, name_dictionary = preprocessor.load_data(state_filter)

	print("District graph has %d nodes and %d edges." % ((len(graph.nodes), len(graph.edges))))

	print("  __________________________")
	print(" /		            \\")
	print("| Gen. Tree Decomposition... |")
	print(" \\__________________________/")
	print()
	
	tree_decomposer = td.Tree_Decomposer(graph)
	nice_tree_decomposition = ntd.Nice_Tree_Decomposition(tree_decomposer.make_nice_tree_decomposition())

	print("NTD has %d nodes" % len(nice_tree_decomposition.graph.nodes))
	print("Nice properties: %r" % tree_decomposer.check_nice_tree_node_properties())

	#print_graph(nice_tree_decomposition, 'output/ntd.png', False)
	#print_graph(nice_tree_decomposition, 'output/ntd_planar.png', True)

	print("  __________________________")
	print(" /		            \\")
	print("|   Starting the Algorithm   |")
	print(" \\__________________________/")
	print()

	algorithm = algo.Algorithm(graph, nice_tree_decomposition, h, k)
	root_node_signature = algorithm.execute()
	
	#print("result:")
	#print(root_node_signature)
	print("  __________________________")
	print(" /	 	            \\")
	print("|  ...Programm is finished   |")
	print(" \\__________________________/")
	
	print(interpret_result(root_node_signature, identifier_to_district_dictionary, position_dictionary, name_dictionary, graph))
	


def interpret_result(root_node_signature, identifier_to_district_dictionary, position_dictionary, name_dictionary, graph):
	min_val = math.inf
	min_edges = set()
	for (edges, val) in root_node_signature.values():
		if (val < min_val):
			min_val = val
			min_edges = edges

	print("Found an edge set to delete with %f edges: %s" % (min_val, min_edges))

	graph.remove_edges_from(min_edges)

	plt.figure(num=None, figsize=(15, 15), dpi=256)
		
	nx.draw_networkx_labels(graph,pos=position_dictionary, labels=name_dictionary,font_size=10)
	nx.draw_networkx_labels(graph,pos=position_dictionary, labels=name_dictionary,font_size=10)
	nx.draw_networkx_nodes(graph, position_dictionary)
	nx.draw_networkx_edges(graph, position_dictionary)
	plt.savefig('output/districtsWithDeletedEdges.png')

	return True

	

if __name__ == "__main__":
	parser = ArgumentParser(add_help=False)
	parser.add_argument("-s", "--state")
	parser.add_argument("-h", type=int)
	parser.add_argument("-k", type=int)
	args = parser.parse_args()

	if (args.h == None):
		print("Error: -h parameter is required!")
		sys.exit(1)

	if (args.k == None):
		print("Error: -k paramter is required!")
		sys.exit(1)

	main(args.h, args.k, args.state)