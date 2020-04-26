import requests
import base64

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=We0w7GgMAQxc9RNQOT0ZZHjh&client_secret=IxfQ20x46RLQlucriQuWbEX4snMfGThb'
response = requests.get(host)
if response:
    print(response.json()['access_token'])


'''
车牌识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
# 二进制方式打开图片文件
f = open('F:\\VS code\\python\\opencv\\carword\\text1.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = str(response.json()['access_token'])
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())