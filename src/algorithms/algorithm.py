import copy

class Algorithm:
	
	def __init__(self, nice_tree_decomposition):
		self.nice_tree_decomposition = nice_tree_decomposition

	def execute(self):
		return True

	def generate_possible_component_states_of_bag(self, bag, size):
		states = []
		all_partitions = self.generate_partitions_of_bag_of_size(bag, size)
		
			
	def generate_partitions_of_bag_of_size(self, bag, size):
		partitions = set()
		initial_partition = Partition([])
		for node in bag:
			initial_partition.blocks.append([])

		partitions.add(initial_partition)

		for node in bag:
			new_partitions = set()
			
			for partition in partitions:
				for block in partition.blocks:
					if len(block) < size:
						
						new_partition = copy.deepcopy(partition)
						# add node to block
						for new_block in new_partition.blocks:
							if new_block == block:
								new_block.append(node)
								break

						add_new_entry = True
						for existing_partition in new_partitions:
							if existing_partition.blocks == new_partition.blocks:
								add_new_entry = False
						if(add_new_entry):
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
	def all_functions_from_partition_to_range(self, partition, h):
		all_functions = set()
		all_functions.add(Function({})) # initial set of undefined functions used to generate the rest

		new_block_list = []
		for block in partition.blocks:
			new_block_list.append(Block(block)) # blocks need to be hashable now so we create a wrapper object
		partition.blocks = new_block_list
		print(partition.blocks)
		
		# partially define function by setting the mapping for each block
		for block in partition.blocks:

			# for each possible value that a block could be mapped to
			new_all_functions = []
			for codomain_value in range(1, h+1):
				# if that value is legal
				if codomain_value >= len(block.node_list):
					# then for each function that already exists
					for function in all_functions:
						# create a new function with the only addition being
						# where that block will be mapped to

						new_function = Function(dict(function.dictionary))
						new_function.dictionary[block] = codomain_value
						new_all_functions.append(new_function)
			
			all_functions = new_all_functions
		for function in all_functions:
			print(function.dictionary)
		return all_functions
			



# We need a wrapper class in order to have hashable lists for the set of Partitions
class Block:
	def __init__(self, node_list):
		self.node_list = node_list

class Partition:
	def __init__(self, blocks):
		self.blocks = blocks

class Function:
	def __init__(self, dictionary):
		self.dictionary = dictionary





