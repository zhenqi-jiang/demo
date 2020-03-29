import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sort_contours(cnts,method="left-to-right"):
    reverse=False
    i=0
    if method=="right-to-left" or method=="bottom-to-top":
        reverse=True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i=1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
    return cnts

#模板文件预处理
moban=cv2.imread("F:\\VS code\\python\\opencv\\images\\ocr_a_reference.png")
shibie=cv2.imread("F:\\VS code\\python\\opencv\\images\\credit_card_04.png")
shibie=cv2.resize(shibie,(583, 368))
show(moban)
mobanhui=cv2.cvtColor(moban,cv2.COLOR_BGR2GRAY)
show(mobanhui)
ret,thresh=cv2.threshold(mobanhui, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
show(thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
moban_img = moban.copy()
cv2.drawContours(moban_img, contours, -1, (0, 0, 255), 2)
show(moban_img)
contours=sort_contours(contours,method="left-to-right")
ku={}
for (i,c) in enumerate(contours):
    x,y,w,h=cv2.boundingRect(contours[i])
    ku_number=thresh[y:y + h, x:x + w]
    ku_number = cv2.resize(ku_number, (57, 88))
    #show(ku_number)
    ku[i]=ku_number

#识别文件预处理
show(shibie)
shibiehui=cv2.cvtColor(shibie,cv2.COLOR_BGR2GRAY)
# show(shibiehui)
#shibie01=cv2.adaptiveThreshold(shibiehui, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 3)
ret,shibie01=cv2.threshold(shibiehui, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show(shibie01)
kernel = np.ones((2,5),np.uint8) 
big = cv2.dilate(shibie01,kernel,iterations = 2)
show(big)
kernel = np.ones((1, 2), np.uint8)
little = cv2.erode(big, kernel,iterations = 2)
show(little)
kernel = np.ones((2,5),np.uint8)
big = cv2.dilate(little,kernel,iterations = 3)
show(big)
contours, hierarchy = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shibie_img=shibie.copy()
cv2.drawContours(shibie_img, contours, -1, (0, 0, 255), 2)
show(shibie_img)
big_img=big.copy()
cv2.drawContours(big_img, contours, -1, (0, 0, 255), 2)
plt.imshow(big_img, 'gray')
plt.show()
zuobiao=[]
for (i,c) in enumerate(contours):
    (x,y,w,h)=cv2.boundingRect(c)
    ar = w / float(h)
    if ar>2.5 and ar<4:
        if w>80 and w<120 and h>30 and h<40:
            zuobiao.append((x,y,w,h))
zuobiao=sorted(zuobiao, key=lambda x:x[0])
print(zuobiao)
result=''
shibie_img=shibie.copy()

#模板匹配
for (i, (gX, gY, gW, gH)) in enumerate(zuobiao):
    ready=shibie01[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    show(ready)
    contours, hierarchy = cv2.findContours(ready, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=sort_contours(contours,method="left-to-right")
    shownumber=''
    for (i,c) in enumerate(contours):
        (x,y,w,h)=cv2.boundingRect(c)
        if h<20 or h>40:
            continue
        copy=ready[y:y + h, x:x + w ]
        copy = cv2.resize(copy, (57, 88))
        show(copy)
        bijiaojieguo={}
        for number in range(10):
            res = cv2.matchTemplate(copy, ku[number], cv2.TM_CCOEFF_NORMED)
            bijiaojieguo[number]=res
        l = sorted(bijiaojieguo.items(), key=lambda d:d[1], reverse=True)
        shownumber=shownumber+str(l[0][0])
        result=result+str(l[0][0])
    #结果展示
    cv2.rectangle(shibie_img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(shibie_img, shownumber, (gX - 15, gY - 15), font, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)


show(shibie_img )
print("输出结果:{}".format(result))



