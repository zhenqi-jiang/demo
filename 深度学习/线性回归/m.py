import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_LABLE=['speallength','sepalwidth','petallength','petalwidth','species']
TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
df_iris=pd.read_csv(train_path,header=0,names=DATA_LABLE)
iris=np.array(df_iris)
plt.figure('iris data',figsize=(15,15))
for i in range(4):
    for j in range(4):
        plt.subplot(4,4,4*i+j+1)
        if(i==j):
            plt.text(0.3,0.5,DATA_LABLE[i],fontsize=15)
        else:
            plt.scatter(iris[:,j],iris[:,i],c=iris[:,4],cmap='brg')
        if(i==0):
            plt.title(DATA_LABLE[j])
        if(j==0):
            plt.ylabel(DATA_LABLE[i])
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()