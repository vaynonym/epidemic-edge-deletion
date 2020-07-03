import networkx as nx
import geojson
import copy
import functools
import os.path



import matplotlib.pyplot as plt

POINT_PROXIMITY_TOLERANCE = 3

FILE_BASE_NAME = "data/graph"

class Preprocessor:

	def __init__(self):
		pass

	def load_data(self, state_filter):
		file = open("data/landkreise-in-germany.geojson", 'r')
		data_dump = geojson.load(file)
		file.close()
		
		if (state_filter == None):
			district_list = [ DistrictPolygon(district)
								for district in data_dump["features"]
								if district.properties["type_2"] != "Water body"]
		else:
			district_list = [ DistrictPolygon(district)
								for district in data_dump["features"]
								if district.properties["type_2"] != "Water body"
									and district.properties["name_1"] in state_filter]
			


		district_graph = nx.Graph()

		name_dictionary = dict()
		position_dictionary = dict()

		identifier_to_district_dictionary = dict()
		identifier = 0

		graph_file_name = FILE_BASE_NAME + "_" + ("_".join(state_filter))

		for district in district_list:
			identifier_to_district_dictionary[identifier] = district
			identifier += 1
		
	

		if os.path.exists(graph_file_name):
			district_graph = load(graph_file_name)
			for i in range(identifier):
				district1 = identifier_to_district_dictionary[i]
				name_dictionary[i] = district1.district_name
				position_dictionary[i] = district1.polygon_coordinate
		else:
			for i in range(identifier):
				district_graph.add_node(i)
				district1 = identifier_to_district_dictionary[i]
				name_dictionary[i] = district1.district_name
				position_dictionary[i] = district1.polygon_coordinate
				for j in range(i + 1, identifier):
					district2 = identifier_to_district_dictionary[j]
					if( district1.do_bounding_boxes_intersect(district2) and district.do_districts_intersect(district2)):
						#if(not district2 in district1.neighbours):
						#	district1.neighbours.append(district2)    
						#if(not district1 in district2.neighbours):
						#	district2.neighbours.append(district1)
						district_graph.add_edge(i, j)
				print("Finished {count} out of {max_count}".format(count = i, max_count = identifier))
			save(district_graph, graph_file_name)

		maxvalue=[0,0]
		for key in position_dictionary:
				position = position_dictionary[key]
				if position[0] > maxvalue[0]:
						maxvalue[0] = position[0]
				if position[1] > maxvalue[1]:
						maxvalue[1] = position[1]

		for key in position_dictionary:
				position = position_dictionary[key]
				position[0] = position[0]/maxvalue[0]
				position[1] = position[1]/maxvalue[1]

		plt.figure(num=None, figsize=(15, 15), dpi=256)

		print(position_dictionary)
		
		nx.draw_networkx_labels(district_graph,pos= position_dictionary, labels=name_dictionary,font_size=10)
		# nx.draw_networkx_labels(district_graph, pos=position_dictionary, font_size=10)
		nx.draw_networkx_nodes(district_graph, pos = position_dictionary)
		nx.draw_networkx_edges(district_graph, pos = position_dictionary)
		plt.savefig('output/districts.png')

		return district_graph, identifier_to_district_dictionary, position_dictionary, name_dictionary

def save(G, fname):
	nx.write_gpickle(G, fname)

def load(fname):
	return nx.read_gpickle(fname)

	

@functools.total_ordering
class DistrictPolygon:
	def __init__(self, district):
		self.id = int(district.properties["cca_2"], 10)
		self.district_name = district.properties["name_2"]
		self.state_name = district.properties["name_1"]
		
		self.polygon_coordinate = district.properties["geo_point_2d"]
		self.polygon_coordinate.reverse() # for some reason the coordinates here seem reversed from all the others...?
		self.geometry = district.geometry
		self.bounding_box = self.calculate_bounding_box()
		self.neighbours = []

	def calculate_bounding_box(self):
		bounding_box = (copy.deepcopy(self.polygon_coordinate), copy.deepcopy(self.polygon_coordinate))

		if self.geometry.type == "Polygon":
			for polygon_part in self.geometry.coordinates:
				for point in polygon_part:
					# top left coordinate
					if(point[0] < bounding_box[0][0]):
						bounding_box[0][0] = point[0]
					if(point[1] < bounding_box[0][1]):
						bounding_box[0][1] = point[1]
					
					# bottom right coordinate
					if(point[0] > bounding_box[1][0]):
						bounding_box[1][0] = point[0]
					if(point[1] > bounding_box[1][1]):
						bounding_box[1][1] = point[1]

		elif self.geometry.type == "MultiPolygon" :
			# account for each Polygon
			for polygon in self.geometry.coordinates:
				# account for holes in polygon
				for polygon_part in polygon:
					for point in polygon_part:
						# top left coordinate
						if(point[0] < bounding_box[0][0]):
							bounding_box[0][0] = point[0]
						if(point[1] < bounding_box[0][1]):
							bounding_box[0][1] = point[1]
						
						# bottom right coordinate
						if(point[0] > bounding_box[1][0]):
							bounding_box[1][0] = point[0]
						if(point[1] > bounding_box[1][1]):
							bounding_box[1][1] = point[1]

		else:
			raise Exception("Type other than Polygon and MultiPolygon")
			
		return bounding_box

	def do_bounding_boxes_intersect(self, other_district_polygon):
		other_bounding_box = other_district_polygon.bounding_box

		bb_top_left = self.bounding_box[0]
		bb_bottom_right = self.bounding_box[1]
		bb_bottom_left = [bb_top_left[0], bb_bottom_right[1]]
		bb_top_right = [bb_bottom_right[0], bb_top_left[1]]

		corners = [bb_bottom_left, bb_bottom_right, bb_top_left, bb_top_right]

		for corner in corners:
			if(		corner[0] >= other_bounding_box[0][0] and corner[0] <= other_bounding_box[1][0]
				and corner[1] >= other_bounding_box[0][1] and corner[1] <= other_bounding_box[1][1]):
				return True
		return False

	def points_are_close(self, point1, point2, abs_tol=0.0):
		return abs(point1[0]-point2[0]) + abs(point1[1] - point2[1]) <= abs_tol

	def do_districts_intersect(self, other_district_polygon):
		if self.geometry.type[0] == "P" and other_district_polygon.geometry.type[0] == "P":
			for polygon_part in self.geometry.coordinates:
				for point in polygon_part:
					for other_polygon_part in other_district_polygon.geometry.coordinates:
						for other_point in other_polygon_part:
							if self.points_are_close(point, other_point, abs_tol=POINT_PROXIMITY_TOLERANCE):
								return True
		elif self.geometry.type[0] == "P" and other_district_polygon.geometry.type[0] == "M":
			for polygon_part in self.geometry.coordinates:
				for point in polygon_part:
					for other_polygon in other_district_polygon.geometry.coordinates:
						# account for holes in polygon
						for other_polygon_part in other_polygon:
							for other_point in other_polygon_part:
								if self.points_are_close(point, other_point, abs_tol=POINT_PROXIMITY_TOLERANCE):
									return True 

		elif self.geometry.type[0] == "M" and other_district_polygon.geometry.type[0] == "P" :
			for polygon in self.geometry.coordinates:
				for polygon_part in self.geometry.coordinates:
					for point in polygon_part:
						for other_polygon_part in other_district_polygon.geometry.coordinates:
							for other_point in other_polygon_part:
								if self.points_are_close(point, other_point, abs_tol=POINT_PROXIMITY_TOLERANCE):
									return True
		elif self.geometry.type[0] == "M" and other_district_polygon.geometry.type[0] == "M":
			for polygon in self.geometry.coordinates:
				for polygon_part in self.geometry.coordinates:
					for point in polygon_part:
						for other_polygon in other_district_polygon.geometry.coordinates:
							# account for holes in polygon
							for other_polygon_part in other_polygon:
								for other_point in other_polygon_part:
									if self.points_are_close(point, other_point, abs_tol=POINT_PROXIMITY_TOLERANCE):
										return True
		else:
			raise Exception("Type other than Polygon and MultiPolygon") 




		
	def __eq__(self, other):
		if not isinstance(other, DistrictPolygon):
			return False
		return self.id == other.id

	def __hash__(self):
		return hash(self.id)
		
	def __lt__(self, other):
		if not isinstance(other, DistrictPolygon):
			return NotImplemented
		return self.id < other.id

