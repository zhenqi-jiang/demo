import tensorflow as tf
import numpy as np

x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
    106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
    62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

learn_rate=0.00001
iter=100

display_step=10

np.random.seed(612)
w=tf.Variable(np.random.randn(),dtype=tf.float64)
b=tf.Variable(np.random.randn(),dtype=tf.float64)

mse=[]

for i in range(0,iter+1):
    
    with tf.GradientTape() as tape:
        pred=w*x+b
        loss=0.5*tf.reduce_mean(tf.square(y-pred))
    mse.append(loss)
    
    dL_dw,dL_db=tape.gradient(loss,[w,b])
    
    w.assign_sub(learn_rate*dL_dw)
    b.assign_sub(learn_rate*dL_db)
    
    if i % display_step==0:
        print('i:%d,loss:%f,w:%f,b:%f'%(i,loss,w.numpy(),b.numpy()))


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
W=tf.Variable(np.random.randn(3,1))

mse=[]

for i in range(0,iter+1):
    
    with tf.GradientTape() as tape:
        PRED=tf.matmul(X,W)
        loss=0.5*tf.reduce_mean(tf.square(Y-PRED))
    mse.append(loss)
    
    dL_dW=tape.gradient(loss,W)
    W.assign_sub(learn_rate*dL_dW)
    
    if i %display_step==0:
        print('i:%d,loss:%f,w:%f,%f,%f'%(i,loss,W[1],W[2],W[0]))

