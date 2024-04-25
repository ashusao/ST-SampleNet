import numpy as np

import osmnx as ox
import pickle

from . import bounding_box_to_grid
from . import get_grid_index


def get_type_tag_dict():
    type_tag_dict = {}

    type_tag_dict['education'] = [{"amenity": "college"}, {"amenity": "university"}, {"amenity": "school"},
                                  {"amenity": "language_school"}, {"amenity": "library"}, {"amenity": "driving_school"},
                                  {"amenity": "research_institute"}, {"amenity": "music_school"}]

    type_tag_dict['transport'] = [{"amenity": "car_rental"}, {"amenity": "car_sharing"}, {"amenity": "car_wash"},
                                  {"amenity": "charging_station"}, {"amenity": "fuel"}, {"amenity": "parking"},
                                  {"amenity": "taxi"}]

    type_tag_dict['recreation'] = [{"amenity": "bar"}, {"amenity": "pub"}, {"amenity": "fast_food"},
                                   {"amenity": "biergarten"}, {"amenity": "cafe"}, {"amenity": "food_court"},
                                   {"amenity": "restaurant"}, {"amenity": "ice_cream"}]

    type_tag_dict['health'] = [ {"amenity": "clinic"}, {"amenity": "dentist"},
                               {"amenity": "doctors"}, {"amenity": "hospital"},
                               {"amenity": "pharmacy"}, {"amenity": "social_facility"}, {"amenity": "veterinary"}]

    type_tag_dict['culture'] = [{"amenity": "arts_centre"},
                                {"amenity": "cinema"}, {"amenity": "community_centre"},
                                {"amenity": "exhibition_centre"}, {"amenity": "fountain"},
                                {"amenity": "nightclub"}, {"amenity": "social_centre"},
                                {"amenity": "stripclub"}, {"amenity": "theatre"}, {"amenity": "place_of_worship"}]

    type_tag_dict['public_service'] = [{"amenity": "courthouse"}, {"amenity": "fire_station"}, {"amenity": "police"},
                                       {"amenity": "post_depot"}, {"amenity": "post_box"},
                                       {"amenity": "post_office"}, {"amenity": "prison"}, {"amenity": "townhall"},
                                       {"amenity": "grave_yard"}]

    type_tag_dict['residential'] = [{"building": "residential"}]

    type_tag_dict['commercial'] = [{"building": "commercial"}, {"building": "industrial"},
                                   {"building": "retail"}, {"building": "warehouse"}, {"building": "supermarket"}]

    type_tag_dict['sports'] = [{'sport': True}]
    type_tag_dict['tourism'] = [{'tourism': True}]

    return type_tag_dict


def extract_and_process_poi(config):
    # boundaries
    min_lat = config.getfloat('boundaries', 'min_lat')
    min_lon = config.getfloat('boundaries', 'min_lon')
    max_lat = config.getfloat('boundaries', 'max_lat')
    max_lon = config.getfloat('boundaries', 'max_lon')

    grid_width = int(config['grid']['width'])
    grid_height = int(config['grid']['height'])
    size = int(config['grid']['size'])

    type_poi_dict = {}
    type_tag_dict = get_type_tag_dict()

    s, n, w, e = min_lat, max_lat, min_lon, max_lon

    for tag_type, tags in type_tag_dict.items():
        result = set()
        for tag in tags:
            # print(tag)
            gdf = ox.features.features_from_bbox(north=n, south=s, east=e, west=w, tags=tag)
            for i, row in gdf.iterrows():
                p = row['geometry'].centroid
                result.add(p)
        print(tag_type, len(result))
        type_poi_dict[tag_type] = list(result)

    grids = bounding_box_to_grid(min_lat, min_lon, max_lat, max_lon, size)

    poi_maps = {}
    for tag_type, points in type_poi_dict.items():
        poi_maps[tag_type] = np.zeros((grid_height, grid_width), dtype=int)
        for point in points:
            idx = get_grid_index(grids, grid_width, grid_height, point.y, point.x)
            if idx == -1:
                continue
            poi_maps[tag_type][idx[0]][idx[1]] += 1

    norm_poi_maps = {}
    for tag_type, poi_map in poi_maps.items():
        org_shape = poi_map.shape
        poi_map = poi_map.flatten()
        norm_map = (poi_map - min(poi_map)) / (max(poi_map) - min(poi_map))
        norm_map = norm_map.reshape(org_shape)
        norm_poi_maps[tag_type] = norm_map

    pois = np.stack(list(norm_poi_maps.values()))

    save_dir = config['data']['dir']
    city = config['data']['city']
    with open(save_dir + '/' + city + '_poi_500m.pkl', 'wb') as f:  # open a text file
        pickle.dump(pois, f)  # serialize the list
    f.close()




