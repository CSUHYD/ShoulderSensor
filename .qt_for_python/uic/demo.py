# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'demo.ui'
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
        Dialog.resize(602, 417)
        self.verticalLayoutWidget = QWidget(Dialog)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(60, 60, 451, 311))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout.addWidget(self.label_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.AA_SN_X = QLabel(self.verticalLayoutWidget)
        self.AA_SN_X.setObjectName(u"AA_SN_X")

        self.horizontalLayout.addWidget(self.AA_SN_X)

        self.AA_SN_Y = QLabel(self.verticalLayoutWidget)
        self.AA_SN_Y.setObjectName(u"AA_SN_Y")

        self.horizontalLayout.addWidget(self.AA_SN_Y)

        self.AA_SN_Z = QLabel(self.verticalLayoutWidget)
        self.AA_SN_Z.setObjectName(u"AA_SN_Z")

        self.horizontalLayout.addWidget(self.AA_SN_Z)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.GH_AA_X = QLabel(self.verticalLayoutWidget)
        self.GH_AA_X.setObjectName(u"GH_AA_X")

        self.horizontalLayout_2.addWidget(self.GH_AA_X)

        self.GH_AA_Y = QLabel(self.verticalLayoutWidget)
        self.GH_AA_Y.setObjectName(u"GH_AA_Y")

        self.horizontalLayout_2.addWidget(self.GH_AA_Y)

        self.GH_AA_Z = QLabel(self.verticalLayoutWidget)
        self.GH_AA_Z.setObjectName(u"GH_AA_Z")

        self.horizontalLayout_2.addWidget(self.GH_AA_Z)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.pushButton = QPushButton(Dialog)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(220, 380, 113, 32))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"AA_SN", None))
        self.AA_SN_X.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.AA_SN_Y.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.AA_SN_Z.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"GH_AA", None))
        self.GH_AA_X.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.GH_AA_Y.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.GH_AA_Z.setText(QCoreApplication.translate("Dialog", u"0.0", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"\u5f00\u59cb", None))
    # retranslateUi

