import sys
import os
import time
import serial
import numpy as np
import torch
from PySide2.QtWidgets import QMainWindow, QApplication
from PySide2.QtCore import Signal, QThread, QCoreApplication
from PySide2.QtGui import QPixmap
from demoUI import Ui_Dialog
from collections import deque
from utils import savgol, get_sensor_scaler
from predict import load_model, device, batch_size, seq_len

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
        self.switchBtnTrainFlag = True


    def signalSlotInit(self):
        self.btnBeginSerial.clicked.connect(self.readSerialThreadSlot)
        self.btnBeginSerial.clicked.connect(self.closebtnBeginSerial)
        
        self.btnInitAngle.clicked.connect(self.openbtnBeginTrain)
        self.btnInitAngle.clicked.connect(self.predictSerialThread.getInitAngle)
        self.btnInitAngle.clicked.connect(self.predictSerialThread.closeUpdateInitLabel)

        self.btnBeginTrain.clicked.connect(self.predictSerialThread.chgUpdateDiffLabel)
        self.btnBeginTrain.clicked.connect(self.switchBtnTrain)

        self.readSerialThread.sinOut.connect(self.predictSerialThread.updtSensor)

        self.predictSerialThread.sinOutInit.connect(self.updateInitAngleLbl)
        self.predictSerialThread.sinUpdtInit.connect(self.updateInitAngleLbl)
        self.predictSerialThread.sinUpdtDiff.connect(self.updateDiffAngleLbl)
        self.predictSerialThread.sinUpdtDiff.connect(self.switchCpsLabel)

    # ----------- Slot ---------------
    def readSerialThreadSlot(self):
        self.readSerialThread.start()
        time.sleep(3)
        self.predictSerialThread.start()

    def openbtnBeginSerial(self):
        self.btnBeginSerial.setEnabled(True)

    def closebtnBeginSerial(self):
        self.btnBeginSerial.setEnabled(False)

    def openbtnBeginTrain(self):
        self.btnBeginTrain.setEnabled(True)

    def closebtnBeginTrain(self):
        self.btnBeginTrain.setEnabled(False)

    def updateInitAngleLbl(self, obj):
        self.Init_AA_SN_X.setText(str(obj[0]))
        self.Init_AA_SN_Y.setText(str(obj[1]))
        self.Init_AA_SN_Z.setText(str(obj[2]))
        self.Init_GH_AA_X.setText(str(obj[3]))
        self.Init_GH_AA_Y.setText(str(obj[4]))
        self.Init_GH_AA_Z.setText(str(obj[5]))

    def updateDiffAngleLbl(self, obj):
        self.Act_AA_SN_X.setText(str(obj[0]))
        self.Act_AA_SN_Y.setText(str(obj[1]))
        self.Act_AA_SN_Z.setText(str(obj[2]))
        self.Act_GH_AA_X.setText(str(obj[3]))
        self.Act_GH_AA_Y.setText(str(obj[4]))
        self.Act_GH_AA_Z.setText(str(obj[5]))

    def switchCpsLabel(self, obj):
        flag = abs(obj) > 5
        cpsLabelList = [self.Cps_AA_SN_X, self.Cps_AA_SN_Y, self.Cps_AA_SN_Z, self.Cps_GH_AA_X, self.Cps_GH_AA_Y, self.Cps_GH_AA_Z]
        for i, switch in enumerate(flag):
            if switch:
                cpsLabelList[i].setPixmap(QPixmap(u"res/cps.png"))
            else:
                cpsLabelList[i].clear()

    def switchBtnTrain(self):
        self.switchBtnTrainFlag = not self.switchBtnTrainFlag
        if self.switchBtnTrainFlag == True:
            self.btnBeginTrain.setText(QCoreApplication.translate("Dialog", u"\u5f00\u59cb\u8bad\u7ec3", None))
        else:
            self.btnBeginTrain.setText(QCoreApplication.translate("Dialog", u"\u505c\u6b62\u8bad\u7ec3", None))

class ReadSerial(QThread):
    sinOut = Signal(object)
    def __init__(self, parent=None):
        super(ReadSerial, self).__init__(parent)
        self.ser = serial.Serial(  # 下面这些参数根据情况修改
        port='/dev/cu.usbserial-1430',  # 串口
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
        deqSensor = deque(maxlen=seq_len)

        while True:
            data = self.ser.readline().decode("utf-8")
            data_list = np.array([np.double(i) for i in data.split(',')[:-1]])
            deqSensor.append(data_list)
            sensor = np.array(list(deqSensor))
            sensor = scaler.fit_transform(sensor)
            self.sinOut.emit(sensor)


class PredictSerial(QThread):
    sinOutInit = Signal(object)
    sinUpdtInit = Signal(object)
    sinUpdtDiff = Signal(object)

    def __init__(self, parent=None):
        super(PredictSerial, self).__init__(parent)
        self.sensor = None
        self.rtmAngle = None
        self.initAngle = None
        self.diffAngle = None
        self.doUpdateInitLabel = True
        self.doUpdateDiffLabel = False

    def __del__(self):
        #线程状态改变与线程终止
        self.working = False
        self.wait()
        print('挂起线程【Predict】')

    def updtSensor(self, sensor):
        self.sensor = sensor

    def updtDiffAngle(self):
        diffAngle = self.rtmAngle - self.initAngle
        diffAngle = np.around(diffAngle, 1)
        return diffAngle
        
    def getInitAngle(self):
        self.initAngle = self.rtmAngle
        self.sinUpdtInit.emit(self.initAngle)

    def closeUpdateInitLabel(self):
        self.doUpdateInitLabel = False

    def chgUpdateDiffLabel(self):
        self.doUpdateDiffLabel = not self.doUpdateDiffLabel

    def run(self):
        print('开始线程【Predict】')
        while True:
            if not((self.sensor is None) or (self.sensor.shape != (seq_len, 5))):
                sensor = self.sensor
                # ## filter
                for i in range(sensor.shape[1]):
                    sensor[:, i] = savgol(sensor[:, i], 51, 2, do_plot=False)
                ## fit batch size
                sensor = np.stack([sensor]*batch_size)
                sensor = torch.from_numpy(sensor).float().to(device)
                ## predict
                angle = lstm(sensor)
                angle = angle[0].data.numpy()
                for i in range(len(angle)):
                    if angle[i] > 90:
                        angle[i] = 180 - angle[i]
                angle = np.around(angle, 1)
                self.rtmAngle = angle
                ## update GUI label
                if self.doUpdateInitLabel:
                    self.sinOutInit.emit(self.rtmAngle)
                if self.doUpdateDiffLabel:
                    self.diffAngle = self.updtDiffAngle()
                    self.sinUpdtDiff.emit(self.diffAngle)
                time.sleep(0.1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
