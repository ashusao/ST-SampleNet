import numpy as np
import pandas as pd
import pyarrow
import pickle
import sys

from functools import partial
import multiprocessing as mp

from shapely.geometry import Point, shape
from shapely.ops import transform
import pyproj
import fiona

from . import bounding_box_to_grid
from . import get_grid_index

def get_lat_lon(taxi_zone):
    content = []
    # transform to 4326
    wgs84 = pyproj.CRS('EPSG:4326')
    init_crs = taxi_zone.crs
    project = pyproj.Transformer.from_crs(init_crs, wgs84, always_xy=True).transform

    for zone in taxi_zone:
        bbox = shape(zone['geometry']).bounds
        loc_id = zone['properties']['OBJECTID']

        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2

        # compute center point
        bbox_c = transform(project, Point(x, y))

        content.append((loc_id, bbox_c.x, bbox_c.y))
    return pd.DataFrame(content, columns=["OBJECTID", "longitude", "latitude"])


def load_data():
    dir = config['data']['dir']
    df_07 = pd.read_parquet( dir + "yellow_tripdata_2023-07.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df_08 = pd.read_parquet(dir + "yellow_tripdata_2023-08.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df_09 = pd.read_parquet(dir + "yellow_tripdata_2023-09.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df_10 = pd.read_parquet(dir + "yellow_tripdata_2023-10.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df_11 = pd.read_parquet(dir + "yellow_tripdata_2023-11.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df_12 = pd.read_parquet(dir + "yellow_tripdata_2023-12.parquet",
                            columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                     'PULocationID', 'DOLocationID'])
    df = pd.concat([df_07, df_08, df_09, df_10, df_11, df_12], ignore_index=True)

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.floor('h')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime']).dt.floor('h')
    df.drop(df[df['tpep_pickup_datetime'] < pd.Timestamp(2023, 7, 1)].index, inplace=True)
    df.drop(df[df['tpep_pickup_datetime'] >= pd.Timestamp(2024, 1, 1)].index, inplace=True)
    df.drop(df[df['tpep_dropoff_datetime'] < pd.Timestamp(2023, 7, 1)].index, inplace=True)
    df.drop(df[df['tpep_dropoff_datetime'] >= pd.Timestamp(2024, 1, 1)].index, inplace=True)
    df.drop(df[df['PULocationID'] >= 264].index, inplace=True)  # drop unknown zones
    df.drop(df[df['DOLocationID'] >= 264].index, inplace=True)  # drop unknown zones

    # transform zone to lat lon
    taxi_zone = fiona.open('/data/shared/sao/NYCTaxi/taxi_zones/taxi_zones.shp')
    # extract properties from shape file
    fields_name = list(taxi_zone.schema['properties'])
    # create dictionary of properties and its values
    shp_attr = [dict(zip(fields_name, zone['properties'].values())) for zone in taxi_zone]
    # create dataframe
    df_loc = pd.DataFrame(shp_attr).join(get_lat_lon(taxi_zone).set_index("OBJECTID"), on="OBJECTID")
    # df_loc = pd.DataFrame(shp_attr)
    df_loc.drop_duplicates(inplace=True)
    print(df_loc.shape)

    locid_long_dict = pd.Series(df_loc.longitude.values, index=df_loc.OBJECTID).to_dict()
    locid_lat_dict = pd.Series(df_loc.latitude.values, index=df_loc.OBJECTID).to_dict()

    df['PU_latitude'] = df['PULocationID'].map(locid_lat_dict)
    df['PU_longitude'] = df['PULocationID'].map(locid_long_dict)
    df['DO_latitude'] = df['DOLocationID'].map(locid_lat_dict)
    df['DO_longitude'] = df['DOLocationID'].map(locid_long_dict)

    df.sort_values(by='tpep_pickup_datetime', inplace=True)
    df = df.reset_index(drop=True)

    print(df.shape)

    return df


def process_dataset(i, df, grid_width, grid_height, grids):

    pickup_counts = {}
    dropoff_counts = {}

    if i % 100000 == 0:
        print(i, sep=' ', end=' ')
        sys.stdout.flush()

    grid_ind = get_grid_index(grids, grid_width, grid_height, df.loc[i, 'PU_latitude'], df.loc[i, 'PU_longitude'])
    if grid_ind != -1:
        pickup_counts[df.loc[i, 'tpep_pickup_datetime'].strftime('%Y-%m-%d %H:%M:00')] = grid_ind

    grid_ind = get_grid_index(grids, grid_width, grid_height, df.loc[i, 'DO_latitude'], df.loc[i, 'DO_longitude'])
    if grid_ind != -1:
        dropoff_counts[df.loc[i, 'tpep_dropoff_datetime'].strftime('%Y-%m-%d %H:%M:00')] = grid_ind

    return pickup_counts, dropoff_counts


def pre_process_nyc(config):

    pickup_counts = dict()
    dropoff_counts = dict()

    grid_width = int(config['grid']['width'])
    grid_height = int(config['grid']['height'])
    quantize = config['quantize']['freq']
    size = int(config['grid']['size'])

    # boundaries
    min_lat = config.getfloat('boundaries', 'min_lat')
    min_lon = config.getfloat('boundaries', 'min_lon')
    max_lat = config.getfloat('boundaries', 'max_lat')
    max_lon = config.getfloat('boundaries', 'max_lon')

    grids = bounding_box_to_grid(min_lat, min_lon, max_lat, max_lon, size)

    timestamps = [ts.strftime('%Y-%m-%d %H:%M:00') for ts in
                  pd.date_range(start='2023-07-01', end='2024-01-01', freq=quantize, inclusive='left')]

    for key in timestamps:
        pickup_counts[key] = np.zeros((grid_height, grid_width), dtype=int)
        dropoff_counts[key] = np.zeros((grid_height, grid_width), dtype=int)

    df = load_data(config)

    workers = partial(process_dataset, df=df, grid_width=grid_width, grid_height=grid_height, grids=grids)

    indices = list(range(len(df)))

    pool = mp.Pool(processes=40)
    results = pool.map(workers, indices, chunksize=100000)

    pool.close()
    pool.join()

    for result in results:
        # update pickup counts
        for ts, grid in result[0].items():
            if ts in pickup_counts:
                if grid[0] < 0 or grid[0] >= grid_height or \
                        grid[1] < 0 or grid[1] >= grid_width:
                    continue
                pickup_counts[ts][grid[0]][grid[1]] += 1

        # update dropoff counts
        for ts, grid in result[1].items():
            if ts in dropoff_counts:
                if grid[0] < 0 or grid[0] >= grid_height or \
                        grid[1] < 0 or grid[1] >= grid_width:
                    continue
                dropoff_counts[ts][grid[0]][grid[1]] += 1

    print('****** Pre Processing Finished ********')

    save_dir = config['data']['dir']
    city = config['data']['city']

    with open(save_dir + '/' + city + '_pickup_500m_' + quantize + '.pkl', 'wb') as f:  # open a text file
        pickle.dump(pickup_counts, f)  # serialize the list
    f.close()

    with open(save_dir + '/'+ city + '_dropoff_500m_' + quantize + '.pkl', 'wb') as f:  # open a text file
        pickle.dump(dropoff_counts, f)  # serialize the list
    f.close()




