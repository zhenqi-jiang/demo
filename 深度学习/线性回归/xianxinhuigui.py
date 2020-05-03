#纯python实现一元线性回归
x=[137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
    106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21]
y=[145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
    62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30]

meanx=sum(x)/len(x)
meany=sum(y)/len(x)

sumxy=0.0
sumx=0.0
for i in range(len(x)):
    sumxy+=(x[i]-meanx)*(y[i]-meany)
    sumx+=(x[i]-meanx)*(x[i]-meanx)

w=sumxy/sumx
b=meany-w*meanx

x_test=[128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00]
print('面积\t价格')
for i in range(len(x_test)):
    print(x_test[i],'\t',round((w*x_test[i]+b),2))


#numpy实现
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

a=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
           106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
b=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
           62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

meana=np.mean(a)
meanb=np.mean(b)

sumab=np.sum((a-meanb)*(b-meanb))
suma=np.sum(pow(a-meana,2))

k=sumab/suma
c=meanb-k*meana

x_test=np.array([128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00])
y_pred=k*x_test+c

print('面积\t价格')
for i in range(len(x_test)):
    print(x_test[i],'\t',round(y_pred[i],2))

plt.figure()

plt.scatter(a,b,color="red",label='销售记录')
plt.scatter(x_test,y_pred,color='blue',label='预测房价')
plt.plot(x_test,y_pred,color='green',label='拟合曲线',linewidth=2)

plt.xlabel('面积',fontproperties='SimHei',fontsize=14)
plt.ylabel('价格',fontproperties='SimHei',fontsize=14)

plt.xlim((40,150))
plt.ylim((40,150))

plt.legend(loc='upper left')
plt.show()

#tensorflow实现
import tensorflow as tf

x=tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
                106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
                62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

meanx=tf.reduce_mean(x)
meany=tf.reduce_mean(y)

sumxy=tf.reduce_sum((x-meanx)*(y-meany))
sumx=tf.reduce_sum(pow(x-meanx,2))

w=sumxy/sumx
b=meany-w*meanx

x_test=tf.constant([128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00])
y_pred=w*x_test+b

print(y_pred)
