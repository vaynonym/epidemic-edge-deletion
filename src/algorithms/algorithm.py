import copy

class Algorithm:
	
	def __init__(self, nice_tree_decomposition):
		self.nice_tree_decomposition = nice_tree_decomposition

	def execute(self):
		return True

	def generate_possible_component_states_of_bag(bag):
		states = []

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

class Partition:
	def __init__(self, blocks):
		self.blocks = blocks





