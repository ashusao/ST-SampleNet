import configparser

from utils.preprocess_nyc_taxi import pre_process_nyc
from utils.poi import extract_and_process_poi

from train import train


if __name__=='__main__':

    # read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    pre_process_flag = config.getboolean('general', 'pre_process')
    train_flag = config.getboolean('general', 'train')
    process_poi = config.getboolean('general', 'process_poi')

    if process_poi:
        extract_and_process_poi(config)

    if pre_process_flag:
        pre_process_nyc(config)

    if train_flag:
        train(config)


