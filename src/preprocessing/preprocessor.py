import networkx as nx
import geojson


class Preprocessor:

	def __init__(self):
		pass

	def load_data(self):
		file = open("data/landkreise-in-germany.geojson", 'r')
		data_dump = geojson.load(file)

		district_list = [ DistrictPolygon(district.properties["name_2"],
										  district.properties["name_1"],
										  district.geometry			    )
									for district in data_dump["features"]] 
												
		graph = nx.Graph()
		return graph
	


class DistrictPolygon:
	def __init__(self, district_name, county_name, polygon_representation):
		self.district_name = district_name
		self.county_name = county_name
		self.polygon_representation = polygon_representation



