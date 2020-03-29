import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


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

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped 
    

def mobanchuli(words,number):
    ku={}
    word_k=0
    number_k=0
    for i in range(1,13):
        moban=cv2.imread('F:\\VS code\\python\\opencv\\carword\\{}.jpg'.format(i))
        #show(moban)
        img_gray = cv2.cvtColor(moban, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1) 
        #show(gaussian)
        thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if i <=5:
            kernel = np.ones((10,2),np.uint8)
            big = cv2.dilate(thresh,kernel,iterations = 4)
            contours, hierarchy = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            moban_img = moban.copy()
            cv2.drawContours(moban_img, contours, -1, (0, 0, 255), 2)
            #show(moban_img)
            contours=sort_contours(contours,method="left-to-right")
            for (a,c) in enumerate(contours):
                x,y,w,h=cv2.boundingRect(contours[a])
                ku_word=thresh[y:y + h, x:x + w]
                ku_word = cv2.resize(ku_word, (60, 120))
                #show(ku_word)
                ku[words[word_k]]=ku_word
                word_k=word_k+1
        elif i==6:
            img_gray = cv2.cvtColor(moban, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1)
            thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ku_word = cv2.resize(thresh, (60, 120))
            #show(ku_word)
            ku[words[word_k]]=ku_word
            word_k=word_k+1
        else:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            moban_img = moban.copy()
            cv2.drawContours(moban_img, contours, -1, (0, 0, 255), 2)
            #show(moban_img)
            contours=sort_contours(contours,method="left-to-right")
            for (a,c) in enumerate(contours):
                x,y,w,h=cv2.boundingRect(contours[a])
                ku_number=thresh[y:y + h, x:x + w]
                ku_number = cv2.resize(ku_number, (60, 120))
                show(ku_number)
                ku[number[number_k]]=ku_number
                number_k=number_k+1
    return ku


words=['京','津','冀','晋','蒙','辽','吉','黑','沪',
    '苏','浙','皖','闽','赣','鲁','豫','鄂','湘',
    '粤','桂','琼','瑜','贵','云','藏','陕',
    '甘','青','宁','新','港','澳','使','领','学',
    '警','川']
number=['A','B','C','D','E','F','G','H',
    'J','K','L','M','N','P','Q','R',
    'S','T','U','V','W','X','Y','Z','1',
    '2','3','4','5','6','7','8','9','0',]
ku=mobanchuli(words,number)
aim=cv2.imread('F:\\VS code\\python\\opencv\\carword\\text4.jpg')
#print(aim.shape)
aim=cv2.resize(aim,(1280,720))
hsv = cv2.cvtColor(aim, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(aim, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1) 
thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
#show(mask)
kernel = np.ones((3,3),np.uint8)
big = cv2.dilate(mask,kernel,iterations = 3)
show(big)
contours, hierarchy = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
docCnt = None

for c in contours:
    if cv2.contourArea(c)>8000:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05*peri, True)
        print(approx)
        if len(approx) == 4:
            docCnt = approx
            break
warped = four_point_transform(aim, docCnt.reshape(4, 2))
warped=cv2.resize(warped,(220,70))
show(warped)
img_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1)
thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel1 = np.ones((2,2),np.uint8)
big = cv2.dilate(opening,kernel1 ,iterations = 2 )
show(big)
contours, hierarchy = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
warped_img=warped.copy()
cv2.drawContours(warped_img,contours,-1,(0,0,255),1)
contours=sort_contours(contours,method="left-to-right")
show(warped_img)
zuobiao=[]
for (i,c) in enumerate(contours):
    (x,y,w,h)=cv2.boundingRect(c)
    ar = h / float(w)
    if ar>1 and ar<3:
        if  h>30 and h<70:
            zuobiao.append((x,y,w,h))
zuobiao=sorted(zuobiao, key=lambda x:x[0])
#print(zuobiao)
answer=''

for (i,(x,y,w,h)) in enumerate(zuobiao):
    ready=warped[y - 2:y + h + 2, x - 2:x + w + 2]
    ready=cv2.resize(ready,(60,120))
    img_gray = cv2.cvtColor(ready, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1)
    thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    show(thresh)
    result={}
    if i==0:
        for word in words:
            res = cv2.matchTemplate(thresh, ku[word], cv2.TM_SQDIFF_NORMED)
            res=abs(res)
            result[word]=res
        l = sorted(result.items(), key=lambda d:d[1], reverse=False)
        answer=answer+l[0][0]
    else:
        for num in number:
            res = cv2.matchTemplate(thresh, ku[num], cv2. TM_SQDIFF_NORMED)
            res=abs(res)
            result[num]=res
        l = sorted(result.items(), key=lambda d:d[1], reverse=False)
        answer=answer+l[0][0]
print(answer)

