import cv2
import requests
import base64
import numpy as np

class deal_image():
    def __init__(self,path):
        self.path=path
        self.result=self.API(self.path)
        self.number=self.result['number']
        self.sensitive_area_image=self.change_image(self.path,self.result)
        self.sensitive_area_01=self.twovalue(self.sensitive_area_image)
        
        
    def API(self,path):
        host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=输入自己的id&client_secret=输入自己的密码'
        response = requests.get(host)
        
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
        f = open(path, 'rb')
        img = base64.b64encode(f.read())
        
        params = {"image":img}
        access_token = str(response.json()['access_token'])
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            return response.json()['words_result']
    
    def change_image(self,path,result):
        image=cv2.imread(path)
        x0=result['vertexes_location'][0]['x']
        y0=result['vertexes_location'][0]['y']
        x1=result['vertexes_location'][1]['x']
        y1=result['vertexes_location'][1]['y']
        x2=result['vertexes_location'][2]['x']
        y2=result['vertexes_location'][2]['y']
        x3=result['vertexes_location'][3]['x']
        y3=result['vertexes_location'][3]['y']
        pts1=np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
        pts2=np.float32([[0,0],[100,0],[100,40],[0,40]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换，参数3是目标图像大小
        dst = cv2.warpPerspective(image, M, (100, 40))
        return dst
    
    def twovalue(self,image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3


#if __name__ == '__main__':
    #path='F:\\VS code\\python\\opencv\\carword\\text1.jpg'
    #result=deal_image(path)
    #cv2.namedWindow('lena2')
    #cv2.imshow('lena2', result.sensitive_area_01)
    #cv2.waitKey(0)