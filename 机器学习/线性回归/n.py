import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (_,_)= boston_housing.load_data(test_split=0)

plt.figure(figsize=(10,10))
for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i],train_y)
    plt.title(i+1)
plt.tight_layout(rect=(0,0,1,0.95))
plt.suptitle('房价关系',fontproperties='SimHei',fontsize=20)
plt.show()
