# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import os
import cv2
import random
import time
from utils.consts import *
from utils.label import auto_label_yolov3
from tools.tracking import tracking
from time import time, sleep

import threading

def create_labelmap(label):

    text = str(labelmap_text) + '\n   name: ' + '"'+label + '"'
    text = text + '\n   label: 1\n   display_name: "' + label + '"' +"\n}"

    file = open(labelmap_path, 'w')
    file.write(text)
    file.close()

def create_txt():
        total_xml = os.listdir(xml_path)
        num=len(total_xml)
        list=range(num)
        tr=int(num*train_percent)
        train=random.sample(list,tr)

        for i  in list:
            name=total_xml[i][:-4]+'\n'
            if i in train:
                ftrain.write(name)
            else:
                ftest.write(name)
        ftrain.close()
        ftest .close()
def read_trainLog():
        """
        get iterations and training loss from training log
        """

        # load all training data
        with open(LogPath) as f:
            data = f.readlines()
            data = data[1:] # remove the first row (title)
            data_len = len(data)
        f.close()
        Iters = [0]
        Accuracy = [0]

        for row in range(data_len):
            data_row = data[row].split(' ') # splitted by ' '
            while '' in data_row:
                data_row.remove('')
            if ("Iteration" in data_row) and (len(data_row)==9):
                tmp = data_row[5][:-1]
                Iters.append(int(tmp))

            if "class1:" in data_row:
                Accuracy.append(float(data_row[-1]))
        return Iters, Accuracy

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(581, 226)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 7, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.train_model)
        self.gridLayout.addWidget(self.pushButton, 8, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.upload)
        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.auto_label)
        self.gridLayout.addWidget(self.pushButton_3, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(479, 0))
        self.label.setMaximumSize(QtCore.QSize(479, 25))
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 6, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setMinimumSize(QtCore.QSize(350, 0))
        self.comboBox.setObjectName("comboBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.comboBox)
        self.comboBox.activated[int].connect(self.chooser)
        self.gridLayout.addLayout(self.formLayout, 2, 0, 1, 1)
        # self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        # self.progressBar_2.setProperty("value", 0)
        # self.progressBar_2.setObjectName("progressBar_2")
        # self.gridLayout.addWidget(self.progressBar_2, 4, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 581, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.progressBar.setFormat(_translate("MainWindow", "%p%"))
        self.pushButton.setText(_translate("MainWindow", "Train Model"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload Video"))
        self.pushButton_3.setText(_translate("MainWindow", "Generate Data"))
        self.label.setText(_translate("MainWindow", "Please upload a video..........."))
        self.label_3.setText(_translate("MainWindow", "Epoch: 0, accuracy: 0"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "Label"))
        self.load_labels()

    def num_examples(self, path):
        return len(os.listdir(path))

    def chooser(self):
        index = self.comboBox.currentIndex()
        if index == 0:
            self.lineEdit.setPlaceholderText("")
            self.lineEdit.setEnabled(False)
           # print('Please choose an object name...')
            self.message("Please choose an object name...")
        elif index == 81:
            #print("we are going tracking...")
            self.lineEdit.setEnabled(True)
            self.lineEdit.setPlaceholderText("Enter object name. (default: NoName)")
        else:
            self.lineEdit.setPlaceholderText("")
            self.lineEdit.setEnabled(False)

    def load_labels(self):
        label_file = open(label_path, 'r')
        labels = label_file.read().split('\n')
        self.comboBox.addItem("")
        self.comboBox.setItemText(0, "Please choose object name...")
        for i in range(1,81,1):
            self.comboBox.addItem("")
            self.comboBox.setItemText(i,labels[i-1])
        self.comboBox.addItem("")
        self.comboBox.setItemText(81, "Other")

    def video2image(self,path):
        '''
        video to image convertor
        '''
        self.label.setText("Uploading file....")
        print(path)
        os.system(copy_command.format(path, video_path))
        if not os.path.exists(save_images):
            os.system('mkdir -p %s'%save_images)
        cap = cv2.VideoCapture(video_path)

        FPS=int(cap.get(cv2.CAP_PROP_FPS))
        random_index = random.sample(range(FPS), FPS)
        indexFrame=random_index[:numFramePerSecond]
        ret = True
        count=0
        numOfImage=0
        while ret:
            ret,frame = cap.read()
            if ret == False:
                break
            count+=1
            if count==FPS:
                count=0
            if count in indexFrame:
                numOfImage+=1
                image_name=video_name+'_%08d'%numOfImage+'.jpg'
                image_path=os.path.join(save_images,image_name)
                cv2.imwrite(image_path,frame)

    def upload(self):
        '''
        this functions is for uploading a video and extract
        images into JPEGImages folder
        '''
        src = QFileDialog.getOpenFileName(None, "Open File")
        src = str(src[0])
        if src=="" or src.split('.')[-1]!='mp4':
            self.label.setText("Please choose a video file format mp4.")
        else:
            self.label.setText(src)
            os.system(rm_cmd.format(src_path))
            self.video2image(src)
            if len(os.listdir(src_path))<min_examples:
                self.label.setText("Please choose a video with longer duration...")
            else:
                self.label.setText("Thanks! Your file is uploaded into server.")

    def auto_label(self):
        '''
        automatic labeling images using tracking method or using trained model
        '''
        label = "NoName"
        index = self.comboBox.currentIndex()

        if self.num_examples(jpg_path)>0:
            os.system(rm_cmd.format(jpg_path))
            os.system(rm_cmd.format(xml_path))

        if self.num_examples(src_path)==0:
            #self.label.setText("Please upload a video for training...")
            self.message("Please upload a video for training...")
        elif index == 0:
            #self.label.setText("Please choose an object name...")
            self.message("Please choose an object name...")
        elif index == 81:
            if str(self.lineEdit.text())!='':
                label = str(self.lineEdit.text())
            self.lineEdit.setEnabled(False)
            self.pushButton_3.setText("Please wait, data generating in process...")
            tracking(label)
            self.say_bye(label)
        else:
            self.pushButton_3.setText("Please wait, data generating in process...")
            label = self.comboBox.currentText()
            auto_label_yolov3(label, index)
            self.say_bye(label)
    # def auto_label_prog(self):
    #     full = self.num_examples(src_path)
    #     sleep(60)
    #     while True:
    #         progress = 100*setProgress()/full
    #         sleep(5)
    #         self.progressBar_2.setValue(progress)
    #         print("Hi")
    # def auto_label(self):
    #     t1 = threading.Thread(target=self.auto_label_func)
    #     t2 = threading.Thread(target=self.auto_label_prog)
    #     # starting thread 1
    #     t1.start()
    #     # starting thread 2
    #    # t2.start()
    #     #sleep(3)
    #     # wait until thread 1 is completely executed
    #     t1.join()
    #    # sleep(1)
    #     # wait until thread 2 is completely executed
    #     #t2.join()
    def say_bye(self, label):
        create_labelmap(label)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.comboBox.setEnabled(False)
        self.message("Congrats, you can start training!!!")
        self.pushButton_3.setText("Data generating is finished!")

    def message(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle("Information!")
        msg.setStandardButtons(QMessageBox.Ok)
        ret = msg.exec_()
        if ret == QMessageBox.Ok:
            return
        msg.show()
    def create_n_start(self):
        #self.message("thread1")
        self.pushButton.setText('Stop Training')
        print("Data processing.")
        self.progressBar.setValue(1)
        create_txt()
        os.system(create_lmdb)
        print("Training started")
        os.system(start_train)


    def update_process(self):
        self.progressBar.setValue(15)
        self.pushButton.setText('Stop Training')
        sleep(20)
        de = 0
        #self.message("thread2 is started")
        while True:
            sleep(3)
            iters, acc = read_trainLog()

            if len(acc)>1:
                de = acc[-1]
            epoch = int(iters[-1]/1000)+1
            self.label_3.setText(progres_text.format(epoch, de))
            progress = 1+ (iters[-1]%1000)/10
            self.progressBar.setValue(progress)




    def train_model(self):
        #self.create_n_start()
        #text = self.
        #self.pushButton.setText("Stop training")

        t1 = threading.Thread(target=self.create_n_start)
        t2 = threading.Thread(target=self.update_process)
        # starting thread 1
        t1.start()
        # starting thread 2
        t2.start()
        sleep(3)
        # wait until thread 1 is completely executed
        t1.join()
        sleep(1)
        # wait until thread 2 is completely executed
        t2.join()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
