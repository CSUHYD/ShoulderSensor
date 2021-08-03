from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler
import numpy as np


def savgol(data, window, k, title='', do_plot=True):
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


if __name__ == '__main__':
    scaler = get_sensor_scaler()
    