import pickle
import geohash

from . import bounding_box_to_grid


def generate_scpe(config):

    min_lat = config.getfloat('boundaries', 'min_lat')
    min_lon = config.getfloat('boundaries', 'min_lon')
    max_lat = config.getfloat('boundaries', 'max_lat')
    max_lon = config.getfloat('boundaries', 'max_lon')

    W = int(config['grid']['width'])
    H = int(config['grid']['height'])
    size = int(config['grid']['size'][:-1])

    grid = bounding_box_to_grid(min_lat, min_lon, max_lat, max_lon, size)

    idx_to_geohash = {}

    for i, g in grid.items():
        lon = g.centroid.x
        lat = g.centroid.y
        idx_to_geohash[i] = [geohash.encode(lon, lat, precision=2), geohash.encode(lon, lat, precision=5),
                             geohash.encode(lon, lat, precision=6),
                             geohash.encode(lon, lat, precision=7)]

    idx_to_geohash_swap = {}
    for i in range(H):
        for j in range(W):
            # bottom row of ST image
            first_row_index = i * W + j

            # top row of ST image
            second_row_index = H * W - (W * (i + 1)) + j

            # Swap values between rows
            idx_to_geohash_swap[first_row_index], idx_to_geohash_swap[second_row_index] = idx_to_geohash[
                                                                                              second_row_index], \
                                                                                          idx_to_geohash[
                                                                                              first_row_index]
    idx_to_geohash_swap = dict(sorted(idx_to_geohash_swap.items()))

    data_dir = config['data']['dir']
    city = city = config['data']['city']

    with open( data_dir + city + '_geohash.pkl', 'wb') as f:  # open a text file
        pickle.dump(idx_to_geohash_swap, f)  # serialize the list
    f.close()
