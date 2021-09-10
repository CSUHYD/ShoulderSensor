from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging


def savgol(data, window, k, title='', do_plot=False):
    result = savgol_filter(data, window, k, mode= 'nearest')
    if do_plot:
        figure(figsize=(16, 9), dpi=100)
        plt.plot(data, color='r', label='raw data')
        plt.plot(result, 'b', label='savgol')
        plt.title(title)
        plt.legend(loc=4)
        plt.savefig(f'result/data_processing/savgol_{title}.png')
        plt.show()

    return result


def get_sensor_scaler(path='data/sensor.npy',):
    with open(path, 'rb') as f:
        sensor = np.load(f)
    scaler = StandardScaler()  # 实例化
    sensor = scaler.fit_transform(sensor)
    print("MEAN and VAR: ", scaler.mean_, scaler.var_)

    return scaler


def minmax_scaler(X, _min=25, _max=225):   
    X_std = (X - _min) / (_max - _min)
    return X_std


def get_logger(LEVEL, log_file = None):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file != None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    scaler = get_sensor_scaler()
    