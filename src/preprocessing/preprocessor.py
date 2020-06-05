import networkx as nx
import geojson
import copy


class Preprocessor:

	def __init__(self):
		pass

	def load_data(self):
		file = open("data/landkreise-in-germany.geojson", 'r')
		data_dump = geojson.load(file)
		file.close()
		
		district_list = [ DistrictPolygon(district)
											for district in data_dump["features"]
											if district.properties["type_2"] != "Water body"]
		
		district_graph = nx.Graph()

		for district1 in district_list:
			district_graph.add_node(district1)
			for district2 in district_list:
				if(not district1 == district2 and district1.do_bounding_boxes_intersect(district2)):
					if(not district2 in district1.neighbours):
						district1.neighbours.append(district2)	
					if(not district1 in district2.neighbours):
						district2.neighbours.append(district1)
					
					district_graph.add_edge(district1, district2)

		return district_graph


	def build_graph(district_list):
		for district in district_list:
			pass
					


	


class DistrictPolygon:

	def __init__(self, district):
				
		self.district_name = district.properties["name_2"]
		self.county_name = district.properties["name_1"]
		
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
		


