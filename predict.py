import cv2
import time
import serial
import numpy as np
import torch
from multiprocessing import Process, Pipe
from collections import deque
from model import LSTM
from utils import savgol, get_sensor_scaler

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
seq_len = 120
inp_dim = 5
mid_dim = 6
num_layers = 2
out_dim = 6


def read_serial(pipe):
    ser = serial.Serial(  # 下面这些参数根据情况修改
        port='/dev/cu.usbserial-1430',  # 串口
        baudrate=9600,  # 波特率
        parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.SEVENBITS
    )
    scaler = get_sensor_scaler()
    data = None
    deqSensor = deque(maxlen=seq_len)
    while True:
      data = ser.readline().decode("utf-8")
      data_list = np.array([np.double(i) for i in data.split(',')[:-1]])
      deqSensor.append(data_list)
      sensor = np.array(list(deqSensor))
      sensor = scaler.fit_transform(sensor)
      pipe.send(sensor)


def predict_serial(pipe):
    print('[INFO] Ready to predict...')
    lstm = load_model()
    lstm.eval()
    for i in range(seq_len):
      sensor = pipe.recv()

    ax = [] 
    plt.figure(figsize=(12, 7), dpi=100)
    while True:
      sensor = pipe.recv()
      ## filter
      for i in range(sensor.shape[1]):
        sensor[:,i] = savgol(sensor[:,i], 51, 2, do_plot=False)
      ## fit batch size
      sensor = np.stack([sensor]*batch_size)
      sensor = torch.from_numpy(sensor).float().to(device)
      ## predict
      outputs = lstm(sensor)
      output = outputs[0].data.numpy()

      # plot
      ax.append(list(output))
      ax_ = np.array(ax)
      plt.clf()       
      plt.plot(ax_[:, 0], label='AA_SN_X')
      plt.plot(ax_[:, 1], label='AA_SN_Y')
      plt.plot(ax_[:, 2], label='AA_SN_Z')
      plt.plot(ax_[:, 3], label='GH_AA_X')
      plt.plot(ax_[:, 4], label='GH_AA_Y')
      plt.plot(ax_[:, 5], label='GH_AA_Z')

      axes = plt.gca()
      axes.set_ylim([0, 180])
      plt.legend()
      plt.savefig('result/test/realtime.png')


def load_model():
    print('[INFO] Load model...')
    lstm = LSTM(batch_size, inp_dim, mid_dim,
                num_layers, out_dim, seq_len).to(device)
    lstm.load_state_dict(torch.load(
        'model/model.pth', map_location=device), strict=False)
    print('[INFO] model loaded successfully!')

    return lstm



if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    pipe = Pipe()
    pw = Process(target=read_serial, args=(pipe[0],))
    pr = Process(target=predict_serial, args=(pipe[1],))
    # 启动子进程pw，写入:
    pw.start()
    time.sleep(3)
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
