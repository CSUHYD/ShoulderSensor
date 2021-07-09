# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'demoUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(842, 522)
        self.horizontalLayoutWidget_3 = QWidget(Dialog)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(210, 460, 421, 51))
        font = QFont()
        font.setFamily(u"Heiti SC")
        self.horizontalLayoutWidget_3.setFont(font)
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setSpacing(30)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_3.setContentsMargins(10, 0, 10, 0)
        self.btnBeginSerial = QPushButton(self.horizontalLayoutWidget_3)
        self.btnBeginSerial.setObjectName(u"btnBeginSerial")
        self.btnBeginSerial.setFont(font)

        self.horizontalLayout_3.addWidget(self.btnBeginSerial)

        self.btnInitAngle = QPushButton(self.horizontalLayoutWidget_3)
        self.btnInitAngle.setObjectName(u"btnInitAngle")
        self.btnInitAngle.setFont(font)

        self.horizontalLayout_3.addWidget(self.btnInitAngle)

        self.btnBeginTrain = QPushButton(self.horizontalLayoutWidget_3)
        self.btnBeginTrain.setObjectName(u"btnBeginTrain")
        self.btnBeginTrain.setFont(font)
        self.btnBeginTrain.setStyleSheet(u"")

        self.horizontalLayout_3.addWidget(self.btnBeginTrain)

        self.gridLayoutWidget = QWidget(Dialog)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(30, 20, 781, 427))
        self.gridLayoutWidget.setFont(font)
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)
        self.label_7.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.label_7.setIndent(-1)
        self.label_7.setOpenExternalLinks(False)

        self.gridLayout.addWidget(self.label_7, 2, 0, 1, 1)

        self.label_4 = QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName(u"label_4")
        font1 = QFont()
        font1.setFamily(u"Heiti SC")
        font1.setPointSize(14)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"")

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        self.label_3 = QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)
        self.label_3.setStyleSheet(u"")

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_6 = QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font1)
        self.label_6.setStyleSheet(u"")

        self.gridLayout.addWidget(self.label_6, 1, 0, 1, 1)

        self.label_9 = QLabel(self.gridLayoutWidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font)
        self.label_9.setStyleSheet(u"background:rgb(249, 249, 249)")

        self.gridLayout.addWidget(self.label_9, 4, 0, 1, 1)

        self.label_13 = QLabel(self.gridLayoutWidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setFont(font)
        self.label_13.setStyleSheet(u"background:rgb(249, 249, 249)")

        self.gridLayout.addWidget(self.label_13, 8, 0, 1, 1)

        self.label_12 = QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font)
        self.label_12.setStyleSheet(u"background:rgb(249, 249, 249)")

        self.gridLayout.addWidget(self.label_12, 7, 0, 1, 1)

        self.label_11 = QLabel(self.gridLayoutWidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)
        self.label_11.setStyleSheet(u"background:rgb(249, 249, 249)")

        self.gridLayout.addWidget(self.label_11, 6, 0, 1, 1)

        self.label_10 = QLabel(self.gridLayoutWidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font1)
        self.label_10.setStyleSheet(u"")

        self.gridLayout.addWidget(self.label_10, 5, 0, 1, 1)

        self.label_8 = QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)
        self.label_8.setStyleSheet(u"background:rgb(249, 249, 249)")

        self.gridLayout.addWidget(self.label_8, 3, 0, 1, 1)

        self.Act_AA_SN_Z = QLabel(self.gridLayoutWidget)
        self.Act_AA_SN_Z.setObjectName(u"Act_AA_SN_Z")
        self.Act_AA_SN_Z.setFont(font)
        self.Act_AA_SN_Z.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_AA_SN_Z.setMargin(10)

        self.gridLayout.addWidget(self.Act_AA_SN_Z, 4, 2, 1, 1)

        self.Init_AA_SN_Y = QLabel(self.gridLayoutWidget)
        self.Init_AA_SN_Y.setObjectName(u"Init_AA_SN_Y")
        self.Init_AA_SN_Y.setFont(font)
        self.Init_AA_SN_Y.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_AA_SN_Y.setMargin(10)

        self.gridLayout.addWidget(self.Init_AA_SN_Y, 3, 1, 1, 1)

        self.Init_AA_SN_Z = QLabel(self.gridLayoutWidget)
        self.Init_AA_SN_Z.setObjectName(u"Init_AA_SN_Z")
        self.Init_AA_SN_Z.setFont(font)
        self.Init_AA_SN_Z.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_AA_SN_Z.setMargin(10)

        self.gridLayout.addWidget(self.Init_AA_SN_Z, 4, 1, 1, 1)

        self.Init_AA_SN_X = QLabel(self.gridLayoutWidget)
        self.Init_AA_SN_X.setObjectName(u"Init_AA_SN_X")
        self.Init_AA_SN_X.setFont(font)
        self.Init_AA_SN_X.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_AA_SN_X.setMargin(10)

        self.gridLayout.addWidget(self.Init_AA_SN_X, 2, 1, 1, 1)

        self.Act_AA_SN_X = QLabel(self.gridLayoutWidget)
        self.Act_AA_SN_X.setObjectName(u"Act_AA_SN_X")
        self.Act_AA_SN_X.setFont(font)
        self.Act_AA_SN_X.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_AA_SN_X.setMargin(10)

        self.gridLayout.addWidget(self.Act_AA_SN_X, 2, 2, 1, 1)

        self.Act_AA_SN_Y = QLabel(self.gridLayoutWidget)
        self.Act_AA_SN_Y.setObjectName(u"Act_AA_SN_Y")
        self.Act_AA_SN_Y.setFont(font)
        self.Act_AA_SN_Y.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_AA_SN_Y.setMargin(10)

        self.gridLayout.addWidget(self.Act_AA_SN_Y, 3, 2, 1, 1)

        self.Init_GH_AA_X = QLabel(self.gridLayoutWidget)
        self.Init_GH_AA_X.setObjectName(u"Init_GH_AA_X")
        self.Init_GH_AA_X.setFont(font)
        self.Init_GH_AA_X.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_GH_AA_X.setMargin(10)

        self.gridLayout.addWidget(self.Init_GH_AA_X, 6, 1, 1, 1)

        self.Act_GH_AA_X = QLabel(self.gridLayoutWidget)
        self.Act_GH_AA_X.setObjectName(u"Act_GH_AA_X")
        self.Act_GH_AA_X.setFont(font)
        self.Act_GH_AA_X.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_GH_AA_X.setMargin(10)

        self.gridLayout.addWidget(self.Act_GH_AA_X, 6, 2, 1, 1)

        self.Act_GH_AA_X_2 = QLabel(self.gridLayoutWidget)
        self.Act_GH_AA_X_2.setObjectName(u"Act_GH_AA_X_2")
        self.Act_GH_AA_X_2.setFont(font)
        self.Act_GH_AA_X_2.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_GH_AA_X_2.setMargin(10)

        self.gridLayout.addWidget(self.Act_GH_AA_X_2, 7, 2, 1, 1)

        self.Act_GH_AA_X_3 = QLabel(self.gridLayoutWidget)
        self.Act_GH_AA_X_3.setObjectName(u"Act_GH_AA_X_3")
        self.Act_GH_AA_X_3.setFont(font)
        self.Act_GH_AA_X_3.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Act_GH_AA_X_3.setMargin(10)

        self.gridLayout.addWidget(self.Act_GH_AA_X_3, 8, 2, 1, 1)

        self.Init_GH_AA_Y = QLabel(self.gridLayoutWidget)
        self.Init_GH_AA_Y.setObjectName(u"Init_GH_AA_Y")
        self.Init_GH_AA_Y.setFont(font)
        self.Init_GH_AA_Y.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_GH_AA_Y.setMargin(10)

        self.gridLayout.addWidget(self.Init_GH_AA_Y, 7, 1, 1, 1)

        self.Init_GH_AA_Z = QLabel(self.gridLayoutWidget)
        self.Init_GH_AA_Z.setObjectName(u"Init_GH_AA_Z")
        self.Init_GH_AA_Z.setFont(font)
        self.Init_GH_AA_Z.setStyleSheet(u"background:rgb(249, 249, 249)")
        self.Init_GH_AA_Z.setMargin(10)

        self.gridLayout.addWidget(self.Init_GH_AA_Z, 8, 1, 1, 1)

        self.label_5 = QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)
        self.label_5.setStyleSheet(u"")

        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.btnBeginSerial.setText(QCoreApplication.translate("Dialog", u"\u5f00\u59cb\u91c7\u96c6", None))
        self.btnInitAngle.setText(QCoreApplication.translate("Dialog", u"\u786e\u8ba4\u521d\u59cb\u89d2\u5ea6", None))
        self.btnBeginTrain.setText(QCoreApplication.translate("Dialog", u"\u5f00\u59cb\u8bad\u7ec3", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"\u5173\u8282/\u52a8\u4f5c", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"\u521d\u59cb\u89d2\u5ea6", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"\u80a9\u80db\u80f8\u5173\u8282", None))
        self.label_9.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.label_13.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.label_12.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.label_11.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.label_10.setText(QCoreApplication.translate("Dialog", u"\u76c2\u80b1\u5173\u8282", None))
        self.label_8.setText(QCoreApplication.translate("Dialog", u"       \u4e0a\u63d0+/\u4e0b\u6c89-:", None))
        self.Act_AA_SN_Z.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_AA_SN_Y.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_AA_SN_Z.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_AA_SN_X.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Act_AA_SN_X.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Act_AA_SN_Y.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_GH_AA_X.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Act_GH_AA_X.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Act_GH_AA_X_2.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Act_GH_AA_X_3.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_GH_AA_Y.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.Init_GH_AA_Z.setText(QCoreApplication.translate("Dialog", u"-", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"\u6d3b\u52a8\u89d2\u5ea6", None))
    # retranslateUi

