import networkx as nx
import numpy as np
import src.preprocessing.preprocessor as prepro
import src.algorithms.tree_decomposition as td
import src.algorithms.algorithm as algo


def main():

	print("  __________________________")
	print(" /		            \\")
	print("|     Running Program...     |")
	print(" \\__________________________/")
	print()

	preprocessor = prepro.Preprocessor()
	graph =  preprocessor.load_data()
	
	tree_decomposer = td.Tree_Decomposer(graph)
	nice_tree_decomposition = tree_decomposer.make_nice_tree_decomposition()

	algorithm = algo.Algorithm(nice_tree_decomposition)
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
