import pandas as pd
import numpy as np
import sys

class STSeries(object):

    def __init__(self, config, start_ts, end_ts, volume=0, inflow=0, outflow=0,  pickup=0, dropoff=0, T=24):
        self.config = config
        self.city = config['data']['city']
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.volume = volume
        self.inflow = inflow
        self.outflow = outflow
        self.pickup = pickup
        self.dropoff = dropoff
        self.len_c = int(config['data']['closeness'])
        self.len_p = int(config['data']['period'])
        self.len_t = int(config['data']['trend'])
        self.n_interval = int(config['data']['T'])

        # step frequency
        self.freq = config['quantize']['freq']
        self.T = T

        if T == 24:
            self.ts = pd.date_range(start_ts, end_ts, freq='1H')
        elif T == 48:
            self.ts = pd.date_range(start_ts, end_ts, freq='30min')
        elif T == 1440:
            self.ts = pd.date_range(start_ts, end_ts, freq='1min')

    def generate_instance(self, i, steps):

        x_ = []
        ts_ = []

        for step in steps:

            # generate ts step size behind i
            ip_ts = None
            if self.freq == '30min':
                ip_ts = self.ts[i] - step * pd.Timedelta(30, unit='min')
            if self.freq == '1H':
                ip_ts = self.ts[i] - step * pd.Timedelta(1, unit='hour')

            #print(step, ip_ts)
            # check if input ts is in dataset
            if ip_ts >= self.ts[0] and ip_ts <= self.ts[-1]:
                if self.city == 'NYC':
                    x_.append([self.pickup[ip_ts.strftime('%Y-%m-%d %H:%M:00')],
                               self.dropoff[ip_ts.strftime('%Y-%m-%d %H:%M:00')]])
                elif self.city == 'hannover' or self.city == 'dresden':
                    x_.append([self.volume[ip_ts.strftime('%Y-%m-%d %H:%M:00')],
                               self.inflow[ip_ts.strftime('%Y-%m-%d %H:%M:00')],
                               self.outflow[ip_ts.strftime('%Y-%m-%d %H:%M:00')]])
                else:
                    print('Wrong City Name')
                    exit()
                ts_.append(ip_ts.strftime('%Y-%m-%d %H:%M:00'))
            else:
                print('Not in range: ', ip_ts)
                sys.exit()
        return x_, ts_

    def generate_series(self):

        closeness_only = self.len_p == 0 and self.len_t == 0
        # Create x, X_p, X_t, Y, ts_Y for each set
        c_step = np.arange(1, self.len_c + 1).tolist()
        p_step = [self.n_interval * j for j in np.arange(1, self.len_p + 1)]
        t_step = [7 * self.n_interval * j for j in np.arange(1, self.len_t + 1)]

        #print(len(c_step))

        i = max(7 * self.T * self.len_t, self.T * self.len_p, self.len_c) if not closeness_only else self.len_c

        x_c = []
        x_p = []
        x_t = []
        y = []

        ts_c = []
        ts_p = []
        ts_t = []
        ts_y = []

        #print(len(self.ts))

        while (i < len(self.ts)):
            if self.city == 'NYC':
                y.append([self.pickup[self.ts[i].strftime('%Y-%m-%d %H:%M:00')],
                          self.dropoff[self.ts[i].strftime('%Y-%m-%d %H:%M:00')]])
            elif self.city == 'hannover' or self.city == 'dresden':
                y.append([self.volume[self.ts[i].strftime('%Y-%m-%d %H:%M:00')],
                          self.inflow[self.ts[i].strftime('%Y-%m-%d %H:%M:00')],
                          self.outflow[self.ts[i].strftime('%Y-%m-%d %H:%M:00')]])
            else:
                print('Wrong City Name')
                exit()

            ts_y.append(self.ts[i].strftime('%Y-%m-%d %H:%M:00'))

            x, t = self.generate_instance(i, c_step)  # (c, 3, 17, 22)
            x_c.append(np.stack(x))
            ts_c.append(t)
            if not closeness_only:
                x, t = self.generate_instance(i, p_step)  # (p, 3, 17, 22)
                x_p.append(np.stack(x))
                ts_p.append(t)
                x, t = self.generate_instance(i, t_step)  # (t, 3, 17, 22)
                x_t.append(np.stack(x))
                ts_t.append(t)
            i += 1

        x_c = np.stack(x_c)
        if not closeness_only:
            x_p = np.stack(x_p)
            x_t = np.stack(x_t)
        y = np.asarray(y)

        #print(x_c.shape, x_p.shape, x_t.shape, y.shape)
        if not closeness_only:
            return x_c, x_p, x_t, y, ts_c, ts_p, ts_t, ts_y
        else:
            return x_c, y, ts_c, ts_y


