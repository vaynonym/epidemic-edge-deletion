import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import src.preprocessing.preprocessor as prepro
import src.algorithms.tree_decomposition as td
import src.algorithms.algorithm as algo
import src.algorithms.nice_tree_decomposition as ntd

def print_graph(graph, filename, planar):
	plt.figure(num=None, figsize=(35, 35), dpi=128)
	if (planar):
		nx.draw_planar(graph, node_size=25)
	else:
		nx.draw(graph, node_size=25)

	plt.savefig(filename)

def main():

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
	graph, identifier_to_district_dictionary = preprocessor.load_data()

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

	h = 3
	k = 3
	algorithm = algo.Algorithm(graph, nice_tree_decomposition, h, k)
	result = algorithm.execute()	
	
	print("result:")
	print(result)
	print("  __________________________")
	print(" /	 	            \\")
	print("|  ...Programm is finished   |")
	print(" \\__________________________/")
	return result

if __name__ == "__main__":
	main()