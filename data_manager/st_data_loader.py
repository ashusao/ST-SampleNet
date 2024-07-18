import numpy as np

from . import read_data, normalize_data, timestamp2vec
from .STSeries import STSeries


def load_XY(config, T, len_test):
    print('loading data ...')

    # generate sets of avail data
    start_ts, end_ts = config['data']['avail_data'].split('#')

    pickup, dropoff, pois = read_data(config)

    print('Pickup:')
    mmn_pickup, norm_pickup = normalize_data(pickup, len_test)
    print('Dropoff:')
    mmn_dropoff, norm_dropoff = normalize_data(dropoff, len_test)

    mmn = [mmn_pickup, mmn_dropoff]
    st_series = STSeries(config=config, start_ts=start_ts, end_ts=end_ts,
                         pickup=norm_pickup, dropoff=norm_dropoff, T=T)

    XC = []
    XP = []
    XT = []
    Y = []

    timestamps_XC = []
    timestamps_XP = []
    timestamps_XT = []
    timestamps_Y = []

    print('Creating ST Series ...')
    x_c, x_p, x_t, y, ts_c, ts_p, ts_t, ts_y = st_series.generate_series()

    XC.append(x_c)
    XP.append(x_p)
    XT.append(x_t)
    Y.append(y)
    timestamps_XC += ts_c
    timestamps_XP += ts_p
    timestamps_XT += ts_t
    timestamps_Y += ts_y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)

    print('timestamp array example:')
    print(timestamps_XC[1], timestamps_XP[1], timestamps_XT[1], timestamps_Y[1])

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    print("XC ts: ", len(timestamps_XC), "XP ts: ", len(timestamps_XP), "XT ts: ", len(timestamps_XT),
          "Y ts:", len(timestamps_Y))

    # if config.getboolean('data', 'ohe_ts'):
    timestamps_XC = [timestamp2vec(ts) for ts in timestamps_XC]
    timestamps_XP = [timestamp2vec(ts) for ts in timestamps_XP]
    timestamps_XT = [timestamp2vec(ts) for ts in timestamps_XT]
    timestamps_Y_str = timestamps_Y
    timestamps_Y = timestamp2vec(timestamps_Y)

    return XC, XP, XT, Y, timestamps_XC, timestamps_XP, timestamps_XT,  timestamps_Y, timestamps_Y_str, mmn, pois


def load_train_data(config, T):

    len_test = int(config['test']['n_days']) * T

    XC, XP, XT, Y, timestamps_XC, timestamps_XP, timestamps_XT,  timestamps_Y, timestamps_Y_str, mmn, pois = \
        load_XY(config, T, len_test)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    TS_c_train, TS_p_train, TS_t_train, TS_Y_train, TS_Y_train_str = \
        timestamps_XC[:-len_test], timestamps_XP[:-len_test], timestamps_XT[:-len_test], timestamps_Y[:-len_test], \
        timestamps_Y_str[:-len_test]

    X_train = [XC_train, XP_train, XT_train]
    meta_train = [TS_c_train, TS_p_train, TS_t_train, TS_Y_train, pois, TS_Y_train_str]

    print('X_train shape:')
    for _X in X_train:
        print(_X.shape, )

    print('Y_train shape:', Y_train.shape)

    return X_train, meta_train, Y_train, mmn


def load_test_data(config, T):

    len_test = int(config['test']['n_days']) * T

    XC, XP, XT, Y, timestamps_XC, timestamps_XP, timestamps_XT,  timestamps_Y, timestamps_Y_str, mmn, pois = \
        load_XY(config, T, len_test)

    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    TS_c_test, TS_p_test, TS_t_test, TS_Y_test, TS_Y_test_str = \
        timestamps_XC[-len_test:], timestamps_XP[-len_test:], timestamps_XT[-len_test:], timestamps_Y[-len_test:], \
        timestamps_Y_str[-len_test:]

    X_test = [XC_test, XP_test, XT_test]
    meta_test = [TS_c_test, TS_p_test, TS_t_test, TS_Y_test, pois, TS_Y_test_str]

    print('X_test shape:')
    for _X in X_test:
        print(_X.shape, )

    print('Y_test shape:', Y_test.shape)

    return X_test, meta_test, Y_test, mmn


def load_data(config):

    T = int(config['data']['T'])

    X_train, meta_train, Y_train, mmn_train = load_train_data(config, T)
    X_test, meta_test, Y_test, mmn_test = load_test_data(config, T)

    return X_train, meta_train, Y_train, X_test, meta_test, Y_test, mmn_train, mmn_test


