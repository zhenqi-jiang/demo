import numpy as np
import cv2

capture = cv2.VideoCapture('f:\VS code\python\opencv\\text.mp4')

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
lower_gree = np.array([50,110,110])
upper_gree = np.array([70,255,255])
lower_red1 = np.array([0,110,110])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([170,110,110])
upper_red2 = np.array([179,255,255])

while(True):
    # 1.捕获视频中的一帧
    ret, frame = capture.read()

    # 2.从BGR转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3.inRange()：介于lower/upper之间的为白色，其余黑色
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_gree, upper_gree)
    mask3 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask4 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask  = mask1+mask2+mask3+mask4

    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(30) == ord('q'):
        break