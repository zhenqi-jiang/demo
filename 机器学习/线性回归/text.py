import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
for i in range(4):
    num=np.random.randint(1,10000)
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(train_x[num],cmap='gray')
    plt.title(train_y[num])
plt.show()
