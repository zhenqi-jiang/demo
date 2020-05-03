import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

#载入数据
TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

TEST_URL="http://download.tensorflow.org/data/iris_test.csv"
test_path=tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

df_iris_train=pd.read_csv(train_path,header=0)
iris_train=np.array(df_iris_train)
df_iris_test=pd.read_csv(test_path,header=0)
iris_test=np.array(df_iris_test)

#数据预处理
train_x=iris_train[:,0:2]
train_y=iris_train[:,4]
test_x=iris_test[:,0:2]
test_y=iris_test[:,4]
x_train=train_x[train_y<2]
y_train=train_y[train_y<2]
x_test=test_x[test_y<2]
y_test=test_y[test_y<2]
num_tarin=len(x_train)
num_test=len(x_test)

#尺度相同，不用归一化，只用中心化
x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_test,axis=0)

#可视化样本
plt.figure()
cm_pt=mpl.colors.ListedColormap(['blue','red'])
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)

#构建数据矩阵
x0_train=np.ones(num_tarin).reshape(-1,1)
X_train=tf.cast(tf.concat((x0_train,x_train),axis=1),tf.float32)
Y_train=tf.cast(y_train.reshape(-1,1),tf.float32)

x0_test=np.ones(num_test).reshape(-1,1)
X_test=tf.cast(tf.concat((x0_test,x_test),axis=1),tf.float32)
Y_test=tf.cast(y_test.reshape(-1,1),tf.float32)


#设置超参数
learn_rate=0.2
iter=360

display_step=30

#模型初始值
np.random.seed(612)
W=tf.Variable(np.random.randn(3,1),dtype=tf.float32)
x_=[-1.5,1.5]
y_=-(W[0]+W[1]*x_)/W[2]
plt.plot(x_,y_,color='red',linewidth=3)
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])

#训练模型
loss_train=[]
acc_train=[]
loss_test=[]
acc_test=[]

for i in range(iter+1):
    with tf.GradientTape() as tape:
        PRED_train=1/(1+tf.exp(-tf.matmul(X_train,W)))
        Loss_train=-tf.reduce_mean(Y_train*tf.math.log(PRED_train)+(1-Y_train)*tf.math.log(1-PRED_train))
        PRED_test=1/(1+tf.exp(-tf.matmul(X_test,W)))
        Loss_test=-tf.reduce_mean(Y_test*tf.math.log(PRED_test)+(1-Y_test)*tf.math.log(1-PRED_test))
        
    accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_train.numpy()<0.5,0.,1.),Y_train),tf.float32))
    loss_train.append(Loss_train)
    acc_train.append(accuracy_train)
    accuracy_test=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_test.numpy()<0.5,0.,1.),Y_test),tf.float32))
    loss_test.append(Loss_test)
    acc_test.append(accuracy_test)
    
    dL_dW=tape.gradient(Loss_train,W)
    W.assign_sub(learn_rate*dL_dW)
    
    if i % display_step==0:
        print('i:%i,trainloss:%f,accuracy:%f,testloss:%f,accuracy:%f'%(i,Loss_train,accuracy_train,Loss_test,accuracy_test))
        y_=-(W[0]+W[1]*x_)/W[2]
        plt.plot(x_,y_)
        
#可视化
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(loss_train,color='blue',label='train')
plt.plot(loss_test,color='red',label='test')
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color='blue',label='train')
plt.plot(acc_test,color='red',label='test')
plt.legend()

#数据可视化,色彩分区
M=500
x1_min,x2_min=x_train.min(axis=0)
x1_max,x2_max=x_train.max(axis=0)
t1=np.linspace(x1_min,x1_max,M)
t2=np.linspace(x2_min,x2_max,M)
m1,m2=np.meshgrid(t1,t2)

m0=np.ones(M*M)
X_mesh=tf.cast(np.stack((m0,m1.reshape(-1),m2.reshape(-1)),axis=1),dtype=tf.float32)
Y_mesh=tf.cast(1/(1+tf.exp(-tf.matmul(X_mesh,W))),dtype=tf.float32)
Y_mesh=tf.where(Y_mesh<0.5,0,1)

n=tf.reshape(Y_mesh,m1.shape)
cm_pt=mpl.colors.ListedColormap(['blue','red'])
cm_bg=mpl.colors.ListedColormap(['#FFA0A0','#A0FFA0'])

plt.figure()
plt.pcolormesh(m1,m2,n,cmap=cm_bg)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)

plt.show()