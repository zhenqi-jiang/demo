import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("F:\\VS code\\python\\opencv\\erzhihua.jpg")
img=img[1300:2300,700:2100]
# 原图中卡片的四个角点
pts1 = np.float32([[7, 99], [1331, 96], [33, 946], [1357, 911]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [700, 0], [0, 500], [700, 500]])

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换，参数3是目标图像大小
dst= cv2.warpPerspective(img, M, (700, 500))
img1 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
img2 = cv2.medianBlur(img1, 3)
et2, img3 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
img4 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
img5 = cv2.morphologyEx(img3, cv2.MORPH_GRADIENT, kernel) 
titles = ['Original', 'cvtColor', 'medianBlur', 'threshold','OPEN','CLOSE']
images = [img, img1, img2, img3,img4,img5]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴

plt.show()