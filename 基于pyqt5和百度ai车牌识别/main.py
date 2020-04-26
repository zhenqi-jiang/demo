import sys
import cv2
import requests
import base64
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from Ui_car_ui import Ui_MainWindow
from image_deal import deal_image

class Demo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()       #视频流
        self.CAM_NUM = 0 
        
        self.setupUi(self)
        self.path=''
        self.slot_init()
        
    def slot_init(self):
        self.quite_button.clicked.connect(self.close)
        self.quite2_button.clicked.connect(self.close)
        self.open_image.clicked.connect(self.getimage)
        self.open_cap_button.clicked.connect(self.open_cap_button_clicked)
        self.timer_camera.timeout.connect(self.show_camera)
        self.shibei_button.clicked.connect(lambda:self.image_process(self.path))
        self.shibei_button.setEnabled(False)
        
    def open_cap_button_clicked(self):
        if self.timer_camera.isActive() == False:   #若定时器未启动
            flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:       #flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.open_cap_button.setText('关闭摄像头')
        else:
            self.timer_camera.stop()  #关闭定时器
            self.cap.release()        #释放视频流
            self.cap_show_label.clear()  #清空视频显示区域
            self.open_cap_button.setText('打开摄像头')
 
    def show_camera(self):
        flag,self.image = self.cap.read()  #从视频流中读取
 
        show = cv2.resize(self.image,(640,480))     #把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
        self.cap_show_label.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage
        self.cap_show_label.setScaledContents(True)
        
    def getimage(self):
        #从C盘打开文件格式（*.jpg *.gif *.png *.jpeg）文件，返回路径
        image_file,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open file','C:\\','Image files (*.jpg *.gif *.png *.jpeg)')
        self.path=image_file
        #设置标签的图片
        self.car_image.setPixmap(QtGui.QPixmap(image_file))
        self.car_image.setScaledContents(True)
        self.shibei_button.setEnabled(True)
    
    def image_process(self,path):
        if path=='':
            self.shibei_button.setEnabled(False)
        else:
            result=deal_image(path)
            show = cv2.resize(result.sensitive_area_image,(100,40))     #把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色
            showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
            self.car_lable_image.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage
            self.car_lable_image.setScaledContents(True)
            show = cv2.resize(result.sensitive_area_01,(100,40))     #把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
            self.car_lable_01.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage
            self.car_lable_01.setScaledContents(True)
            self.text_resulat.setText(result.number)
        
    
    def closeEvent(self, QCloseEvent): 
        choice = QtWidgets.QMessageBox.question(self, '退出', '你要退出程序吗?',
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        if choice == QtWidgets.QMessageBox.Yes:
            QCloseEvent.accept()
        elif choice == QtWidgets.QMessageBox.No:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())