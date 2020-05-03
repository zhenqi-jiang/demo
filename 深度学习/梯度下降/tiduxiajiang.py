#numpy实现梯度下降
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
           106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
           62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

learn_rate=0.00001
iter=100

display_step=10

np.random.seed(612)
w=np.random.randn()
b=np.random.randn()

mse=[]

for i in range(0,iter+1):
    
    dL_dw=np.mean(x*(w*x+b-y))
    dL_db=np.mean(w*x+b-y)
    
    w=w-learn_rate*dL_dw
    b=b-learn_rate*dL_db
    
    pred=w*x+b
    loss=np.mean(np.square(y-pred))/2
    mse.append(loss)
    
    if i% display_step==0:
        print("i:%i,loss:%f,w:%f,b:%f" %(i,mse[i],w,b))

plt.figure(figsize=(20,4))

plt.subplot(1,3,1)
plt.scatter(x,y,color='red',label='销售记录')
plt.plot(x,pred,color='blue',label='梯度下降预测')
plt.plot(x,0.89*x+5.41,color='green',label='解析法')
plt.xlabel('面积',fontsize=14)
plt.ylabel('价格',fontsize=14)
plt.legend(loc='upper left')

plt.subplot(1,3,2)
plt.plot(mse)
plt.xlabel('迭代次数',fontsize=14)
plt.ylabel('均方差损失函数',fontsize=14)

plt.subplot(1,3,3)
plt.plot(y,color='red',marker='o',label='销售记录')
plt.plot(pred,color='blue',marker='o',label='梯度下降法')
plt.plot(0.89*x+5.41,color='green',marker='o',label='解析法')
plt.xlabel('样本',fontsize=14)
plt.ylabel('价格',fontsize=14)
plt.legend(loc='upper left')

plt.show()