import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

x_train=train_x[:,5]
y_train=train_y

x_test=test_x[:,5]
y_test=test_y

learn_rate=0.04
iter=2000
display_step=200

np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

mse_train=[]
mse_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        pred_train=w*x_train+b
        loss_train=0.5*tf.reduce_mean(tf.square(y_train-pred_train))
        
        pred_test=w*x_test+b
        loss_test=0.5*tf.reduce_mean(tf.square(y_test-pred_test))
        
    mse_test.append(loss_test)
    mse_train.append(loss_train)
    
    dL_dw,dL_db=tape.gradient(loss_train,[w,b])
    w.assign_sub(learn_rate*dL_dw)
    b.assign_sub(learn_rate*dL_db)
    
    if i%display_step==0:
        print('i:%d,train_loss:%f,test_loss:%f' %(i,loss_train,loss_test))
        
plt.figure(figsize=(15,10))
plt.rcParams['font.sans-serif']=['SimHei']

plt.subplot(2,2,1)
plt.scatter(x_train,y_train,color='blue',label='数据')
plt.plot(x_train,pred_train,color='red',label='预测')
plt.legend(loc='upper left')

plt.subplot(2,2,2)
plt.plot(mse_train,color='blue',linewidth=3,label='训练损失')
plt.plot(mse_test,color='red',linewidth=1.5,label='测试损失')
plt.legend()

plt.subplot(2,2,3)
plt.plot(y_train,color='blue',marker='o',label='真实价格')
plt.plot(pred_train,color='red',marker='*',label='预测')
plt.legend()

plt.subplot(2,2,4)
plt.plot(y_test,color='blue',marker='o',label='真实价格')
plt.plot(pred_test,color='red',marker='*',label='预测')
plt.legend()

plt.show()