import numpy as np 
import pandas as pd 
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#多元线性回归分析
Y=np.array([450,507.7,613.9,563.4,501.5,781,541.8,611.1,1222.1,
            793.2,660.8,792.7,580.8,612.7,890.8,1121,1094.2,1253])
X=np.array([1027.2,1045.2,1225.8,1312.2,1316.4,1442.4,1641,1768.8,1981.2,
            1998.6,2196,2105.4,2147.4,2154,2231.4,2611.8,3143.4,3624.6])
T=np.array([8,9,12,9,7,15,9,10,18,14,10,12,8,10,14,18,16,20])

d=np.stack((X,T,Y),axis=1)
data=pd.DataFrame(d,columns=['X','T','Y'])
print(data)

lm=ols('Y~X+T',data).fit()
print(lm.summary())

#主成分分析
gas1=np.array([0.056,0.049,0.038,0.034,0.084,0.064,0.048,0.069])
gas2=np.array([0.084,0.055,0.130,0.095,0.066,0.072,0.089,0.087])
gas3=np.array([0.031,0.010,0.079,0.058,0.029,0.100,0.062,0.027])
gas4=np.array([0.038,0.110,0.170,0.160,0.320,0.210,0.260,0.250])
gas5=np.array([0.008,0.022,0.058,0.200,0.012,0.028,0.038,0.045])
gas6=np.array([0.022,0.007,0.043,0.029,0.041,0.038,0.036,0.021])

GAS=np.stack((gas1,gas2,gas3,gas4,gas5,gas6),axis=1)
scaler = preprocessing.StandardScaler().fit(GAS)
X_scaler=pd.DataFrame(scaler.transform(GAS),columns=['Z1','Z2','Z3','Z4','Z5','Z6'])
pca=PCA(n_components=6)
pca.fit(X_scaler)
print(pca.explained_variance_ratio_)
print(pca.components_)
result=[]
for i in range(6):
    k=pca.explained_variance_ratio_[:i+1]
    result.append(k.sum())
print(result)

#线性判别分析
Lable=np.array([0,0,0,0,1,1,1,1,2,2,2,2])
GDP=np.array([41890,29461,23381,29663,6000,9060,8402,8677,1128,2299,2370,3071])
age=np.array([77.9,79.1,78.9,79.4,77.7,71.9,71.7,69.6,46.5,49.8,64.6,73.7])
word=np.array([99.5,99.2,96,92.5,99.8,97.3,88.6,92.3,69.1,67.9,49.9,90.3])
study_rate=np.array([93.3,88,99,87.3,87.6,76.8,87.5,71.2,56.2,62.3,40,63.9])
country=['美国','德国','希腊','新加坡','古巴','罗马尼亚','巴西','泰国','尼日利亚','喀麦隆','巴基斯坦','越南']
point=['人均GDP','预期寿命','成人识字率','义务教育入学率']

d=np.stack((GDP,age,word,study_rate),axis=1)
data=pd.DataFrame(d,columns=point)

print(data)

LDA = LinearDiscriminantAnalysis(solver= 'eigen',n_components=1)
LDA.fit(data,Lable.reshape(-1,1))

print('Coefficients:%s, intercept %s'%(LDA.coef_,LDA.intercept_))

