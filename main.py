import configparser

from utils.scpe import generate_scpe
from train import train


if __name__=='__main__':

    # read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    scpe_flag = config.getboolean('general', 'process_scpe')
    train_flag = config.getboolean('general', 'train')

    if scpe_flag:
        generate_scpe(config)

    if train_flag:
        train(config)


