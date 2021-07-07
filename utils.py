from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def savgol(data, window, k, do_plot=True):
    result = savgol_filter(data, window, k, mode= 'nearest')
    if do_plot:
        figure(figsize=(16, 9), dpi=100)
        plt.plot(data, color='r')
        plt.plot(result, 'b', label = 'savgol')
        plt.show()

    return result
