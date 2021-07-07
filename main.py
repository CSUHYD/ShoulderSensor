import time
import random
import serial
import numpy as np
import torch
from multiprocessing import Process, Queue
from collections import deque
from multiprocessing import Queue, Process
from model import LSTM
from utils import savgol


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
seq_len = 10
inp_dim = 5
mid_dim = 6
num_layers = 2
out_dim = 6

def read_serial(deq):
    ser = serial.Serial(  # 下面这些参数根据情况修改
        port='/dev/cu.usbserial-1440',  # 串口
        baudrate=9600,  # 波特率
        parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.SEVENBITS
    )

    data = None
    deqSensor = deque(maxlen=10)
    while True:
      data = ser.readline().decode("utf-8")
      data_list = np.array([np.double(i) for i in data.split(',')[:-1]])
      deqSensor.append(data_list)
      deq.put(deqSensor)


def predict_serial(q):
    print('[INFO] Ready to predict...')
    lstm = load_model()
    for i in range(seq_len):
      value = q.get()
    # while True:
    res = list()
    start = time.time()
    for i in range(1000):
      print(i)
      value = q.get()
      sensor = np.array(list(value))
      # filter
      for i in range(sensor.shape[1]):
        sensor[:,i] = savgol(sensor[:,i], 51, 2, do_plot=False)
      # fit batch size
      sensor = np.stack([sensor]*batch_size)
      sensor = torch.from_numpy(sensor).float().to(device)
      # print(sensor.size())
      # predict
      outputs = lstm(sensor)
      # print(outputs[0])
      end = time.time()
    res.append(end-start)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f" % (time_sum/len(res)))
      

def load_model():
    print('[INFO] Load model...')
    lstm = LSTM(batch_size, inp_dim, mid_dim,
                num_layers, out_dim, seq_len).to(device)
    lstm.load_state_dict(torch.load(
        'model.pth', map_location=device), strict=False)
    print('[INFO] model loaded successfully!')
    lstm.eval()

    return lstm


if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue(maxsize=10)
    pw = Process(target=read_serial, args=(q,))
    pr = Process(target=predict_serial, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    time.sleep(3)
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
