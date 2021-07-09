import sys
import os
import time
import serial
import numpy as np
import torch
from multiprocessing import Process, Pipe
from PySide2.QtWidgets import QMainWindow, QApplication
from PySide2.QtCore import QObject, Signal, Slot, QThread
from demoUI import Ui_Dialog
from collections import deque
from utils import savgol, get_sensor_scaler
from predict import load_model, device, batch_size

os.environ['QT_MAC_WANTS_LAYER'] = '1'
lstm = load_model()
lstm.eval()


class MainWindow(QMainWindow, Ui_Dialog):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("App")
        #实例化多线程对象
        self.readSerialThread = ReadSerial()
        self.predictSerialThread = PredictSerial()
        self.signalSlotInit()

    def signalSlotInit(self):
        self.btnBeginSerial.clicked.connect(self.readSerialThreadSlot)
        self.btnInitAngle.clicked.connect(self.predictSerialThread.getInitAngle)
        self.btnInitAngle.clicked.connect(self.predictSerialThread.closeUpdateLabel)
        self.readSerialThread.sinOut.connect(self.predictSerialThread.updateSensor)
        self.predictSerialThread.sinOut.connect(self.updateInitAngleLbl)
        self.predictSerialThread.sinUpdtInit.connect(self.updateInitAngleLbl)


    # ----------- Slot ---------------
    def readSerialThreadSlot(self):
        self.readSerialThread.start()
        time.sleep(3)
        self.predictSerialThread.start()
        self.btnBeginSerial.setEnabled(False)


    def updateInitAngleLbl(self, obj):
        self.Init_AA_SN_X.setText(str(obj[0]))
        self.Init_AA_SN_Y.setText(str(obj[1]))
        self.Init_AA_SN_Z.setText(str(obj[2]))
        self.Init_GH_AA_X.setText(str(obj[3]))
        self.Init_GH_AA_Y.setText(str(obj[4]))
        self.Init_GH_AA_Z.setText(str(obj[5]))


class ReadSerial(QThread):
    sinOut = Signal(object)

    def __init__(self, parent=None):
        super(ReadSerial, self).__init__(parent)
        self.ser = serial.Serial(  # 下面这些参数根据情况修改
        port='/dev/cu.usbserial-1440',  # 串口
        baudrate=9600,  # 波特率
        parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.SEVENBITS)
        
    def __del__(self):
        self.wait()
        print('挂起线程【Read Serial】')

    def run(self):
        print('开始线程【Read Serial】')
        scaler = get_sensor_scaler()
        data = None
        deqSensor = deque(maxlen=10)

        while True:
            data = self.ser.readline().decode("utf-8")
            data_list = np.array([np.double(i) for i in data.split(',')[:-1]])
            deqSensor.append(data_list)
            sensor = np.array(list(deqSensor))
            sensor = scaler.fit_transform(sensor)
            self.sinOut.emit(sensor)


class PredictSerial(QThread):
    sinOut = Signal(object)
    sinUpdtInit = Signal(object)

    def __init__(self, parent=None):
        super(PredictSerial, self).__init__(parent)
        self.sensor = None
        self.rtmAngle = None
        self.initAngle = None
        self.doUpdateLabel = True

    def __del__(self):
        #线程状态改变与线程终止
        self.working = False
        self.wait()
        print('挂起线程【Predict】')

    def updateSensor(self, sensor):
        self.sensor = sensor

    def getInitAngle(self):
        self.initAngle = self.rtmAngle
        self.sinUpdtInit.emit(self.initAngle)

    def closeUpdateLabel(self):
        self.doUpdateLabel = False

    def run(self):
        print('开始线程【Predict】')
        while True:
            if not((self.sensor is None) or (self.sensor.shape != (10, 5))):
                sensor = self.sensor
                # ## filter
                for i in range(sensor.shape[1]):
                    sensor[:, i] = savgol(sensor[:, i], 51, 2, do_plot=False)
                ## fit batch size
                sensor = np.stack([sensor]*batch_size)
                sensor = torch.from_numpy(sensor).float().to(device)
                ## predict
                angle = lstm(sensor)
                angle = np.around(angle[0].data.numpy(), 1)
                self.rtmAngle = angle
                ## update GUI label
                if self.doUpdateLabel:
                    self.sinOut.emit(self.rtmAngle)
                time.sleep(0.1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
