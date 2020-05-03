import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

area=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
    106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
room=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
price=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
    62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
mun=len(area)

x0=np.ones(mun)
x1=(area-area.min())/(area.max()-area.min())
x2=(room-room.min())/(room.max()-room.min())

X=np.stack((x0,x1,x2),axis=1)
Y=price.reshape(-1,1)

learn_rate=0.02
iter=500

display_step=50

np.random.seed(612)
W=np.random.randn(3,1)

mse=[]

for i in range(0,iter+1):
    dL_dw=np.matmul(np.transpose(X),np.matmul(X,W)-Y)/mun
    W=W-learn_rate*dL_dw
    
    PRED=np.matmul(X,W)
    loss=np.mean(np.square(Y-PRED))/2
    mse.append(loss)
    
    if i%display_step==0:
        print('i:%d,loss:%f'%(i,mse[i]))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(mse)
plt.xlabel('迭代次数',fontsize=14)
plt.ylabel('损失函数',fontsize=14)

plt.subplot(1,2,2)
plt.plot(price,color='red',marker='o',label='销售记录')
plt.plot(PRED,color='blue',marker='*',label='预测房价')
plt.xlabel('样本',fontsize=14)
plt.ylabel('房价',fontsize=14)

plt.legend()
plt.show()