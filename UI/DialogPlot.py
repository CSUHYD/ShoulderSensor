# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DialogPlot.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from collections import deque
import time
matplotlib.use("Qt5Agg")


class MyFigureCanvas(FigureCanvas):
    '''
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    '''
    def __init__(self, parent=None, width=10, height=5, xlim=(0, 2000), ylim=(-2, 2), dpi=100):
        # 创建一个Figure
        fig = plt.Figure(figsize=(width, height), dpi=dpi,
                         tight_layout=True)  # tight_layout: 用于去除画图时两边的空白
        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)
        # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.axes = fig.add_subplot(111)
        self.axes.spines['top'].set_visible(False)  # 去掉上面的横线
        self.axes.spines['right'].set_visible(False)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)


class Ui_DialogPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName(u"DialogPlot")
        self.resize(1320, 760)
        self.graphicsView = QGraphicsView(self)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(20, 20, 1280, 720))
        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self)
        self.colorsSensor = plt.cm.rainbow(np.linspace(0, 1, 5))
        # 初始化 gv_visual_data 的显示
        self.gv_visual_data_content = MyFigureCanvas(width=self.graphicsView.width() / 101,
                                                     height=self.graphicsView.height() / 101,
                                                     xlim=(0, 1000),
                                                     ylim=(0, 250),
                                                     )  # 实例化一个FigureCanvas
        self.dataForPlot = DataForPlot()
        self.refreshPlotInterval = 100        # 每间隔100个数据刷新绘图窗口
        self.sleepPlotTime = self.refreshPlotInterval
        self.angleLabelNames = ['AA_SN_X', 'AA_SN_Y', 'AA_SN_Z', 'GH_AA_X', 'GH_AA_Y', 'GH_AA_Z']
        self.create_graphics_view()

    def retranslateUi(self, DialogPlot):
        DialogPlot.setWindowTitle(QCoreApplication.translate("DialogPlot", u"Dialog", None))

    def create_graphics_view(self):
        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.graphic_scene.addWidget(self.gv_visual_data_content)
        # 把QGraphicsScene放入QGraphicsView
        self.graphicsView.setScene(self.graphic_scene)
        self.graphicsView.show()  # 调用show方法呈现图形

    def plot_sensor_and_angle(self):
        self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
        self.gv_visual_data_content.axes.set_title('Sensors and Angles Monitor')
        self.gv_visual_data_content.axes.set_xlim((0, 1000))
        self.gv_visual_data_content.axes.set_ylim((0, 250))
        self.gv_visual_data_content.axes.plot(self.dataForPlot.sensorNpy, linewidth=1, color='gray', label='Sensor')
        for i in range(6):
            self.gv_visual_data_content.axes.plot(self.dataForPlot.angleNpy[:, i], linewidth=2, label=self.angleLabelNames[i])
        self.gv_visual_data_content.axes.legend()
        self.gv_visual_data_content.draw()  # 刷新画布显示图片，否则不刷新显示

    def recv_sensor_and_angle(self, results):
        sensor = results[0]
        angle = results[1]
        self.dataForPlot.addSensor(sensor)
        self.dataForPlot.addAngle(angle)
        self.dataForPlot.updtNumData()
        if (self.dataForPlot.numData % self.refreshPlotInterval == 0):
            self.do_plot()

    def do_plot(self):
        QTimer.singleShot(self.sleepPlotTime, self.plot_sensor_and_angle)


class DataForPlot(object):
    def __init__(self) -> None:
        super().__init__()
        self.sensor = deque(maxlen=1000)
        self.angle = deque(maxlen=1000)
        self.sensorNpy = np.array(self.sensor)
        self.angleNpy = np.array(self.angle)
        self.numData = 0
    
    def updtNumData(self):
        self.numData += 1
    
    def clearNumData(self):
        self.numData = 0

    def updtSensorNpy(self):
        self.sensorNpy = np.array(self.sensor)

    def updtAngleNpy(self):
        self.angleNpy = np.array(self.angle)

    def addSensor(self, sensor):
        self.sensor.append(sensor)
        self.updtSensorNpy()

    def addAngle(self, angle):
        self.angle.append(angle)
        self.updtAngleNpy()
