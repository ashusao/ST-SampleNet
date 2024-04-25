import numpy as np
import pandas as pd
import pickle
import math
import os
import collections

from sklearn.preprocessing import MinMaxScaler


def hann_dresden_f_name(config):
    f_vol = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_vol_' + str(config['grid']['size']) +
                         '_' + str(config['quantize']['freq'] + '.pkl'))

    f_inflow = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_inflow_' + str(config['grid']['size']) +
                         '_' + str(config['quantize']['freq'] + '.pkl'))

    f_outflow = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_outflow_' + str(config['grid']['size']) +
                         '_' + str(config['quantize']['freq'] + '.pkl'))

    return f_vol, f_inflow, f_outflow


def nyc_f_name(config):
    f_pickup = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_pickup_' + str(config['grid']['size']) +
                         '_' + str(config['quantize']['freq'] + '.pkl'))

    f_dropoff = os.path.join(config['data']['dir'],
                         config['data']['city'] + '_dropoff_' + str(config['grid']['size']) +
                         '_' + str(config['quantize']['freq'] + '.pkl'))


    return f_pickup, f_dropoff


def load_pickle(f_name):
    with open(f_name, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def read_data(config):

    city = config['data']['city']
    pois = load_pickle(config['data']['dir'] + '/' + city + '_poi_500m.pkl')

    if city == 'hannover' or city == 'dresden':
        f_vol, f_inflow, f_outflow = hann_dresden_f_name(config)

        volume = collections.OrderedDict(sorted(load_pickle(f_vol).items()))
        inflow = collections.OrderedDict(sorted(load_pickle(f_inflow).items()))
        outflow = collections.OrderedDict(sorted(load_pickle(f_outflow).items()))

        return volume, inflow, outflow, pois
    elif city == 'NYC':
        f_pickup, f_dropoff = nyc_f_name(config)

        pickup = collections.OrderedDict(sorted(load_pickle(f_pickup).items()))
        dropoff = collections.OrderedDict(sorted(load_pickle(f_dropoff).items()))

        return pickup, dropoff, pois
    else:
        print('Wrong City Name')
        exit()

def normalize_data(data, len_test):

    keys = data.keys()
    stacked_data = np.stack(list(data.values()))
    print(stacked_data.shape)

    data_train = stacked_data[:-len_test]
    print(data_train.shape, data_train.reshape(-1, 1).shape)

    mmn = MinMaxScaler(feature_range=(-1, 1))
    mmn.fit(data_train.reshape(-1, 1))
    org_shape = stacked_data.shape
    print("min:", mmn.data_min_, "max:", mmn.data_max_)

    normalized_data = mmn.transform(stacked_data.reshape(-1, 1)).reshape(org_shape)
    normalized_data = collections.OrderedDict(zip(keys, list(normalized_data)))

    return mmn, normalized_data


def cyclic_encode_time_of_day(time):
    minutes_since_midnight = time.hour * 60 + time.minute
    total_minutes_in_a_day = 24 * 60
    angle = 2 * math.pi * minutes_since_midnight / total_minutes_in_a_day
    return math.sin(angle), math.cos(angle)


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = []

    for timestamp in timestamps:
        timestamp = pd.Timestamp(timestamp)

        dow = timestamp.day_of_week
        v = [0 for _ in range(7)] # day of week
        v[dow] = 1
        if dow >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday

        # Encode time from timestamp
        time = timestamp.time()
        time_encode_vector = cyclic_encode_time_of_day(time)

        v += time_encode_vector

        vec.append(v)

    return np.asarray(vec)
