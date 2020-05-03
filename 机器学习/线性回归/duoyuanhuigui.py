#多元线性回归numpy实现
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,
    106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x2=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,
    62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

x0=np.ones(len(x1))
x=np.stack((x0,x1,x2),axis=1)
y=np.array(y).reshape(-1,1)

xt=np.transpose(x)
xtx_1=np.linalg.inv(np.matmul(xt,x))
xtx_1xt=np.matmul(xtx_1,xt)
w=np.matmul(xtx_1xt,y)

w=w.reshape(-1)
print('多元回归方程：y={:.2f}*x1+{:.2f}*x2+{:.2f}'.format(w[1],w[2],w[0]))
print('请输入房屋面积和房间数')
#x1_test=float(input('房屋面积：'))
#x2_test=int(input('房间数：'))

#y_pred=w[1]*x1_test+w[2]*x2_test+w[0]
#print('预测结果：',round(y_pred,2),'万元')

X1,X2=np.meshgrid(x1,x2)
Y_pred=w[0]+w[1]*X1+w[2]*X2

fig=plt.figure(figsize=(8,6))
ax3d=Axes3D(fig)
#ax3d.view_init(elev=0,azim=0)

#ax3d.plot_surface(X1,X2,Y_pred,cmap='coolwarm')
ax3d.scatter(x1,x2,y,color='b',marker='*')
ax3d.scatter(X1,X2,Y_pred,color='r')
ax3d.plot_wireframe(X1,X2,Y_pred,color='c',linewidth=0.5,)

ax3d.set_xlabel('area',color='r',fontsize=16)
ax3d.set_ylabel('room',color='r',fontsize=16)
ax3d.set_zlabel('price',color='r',fontsize=16)
ax3d.set_yticks([1,2,3])
ax3d.set_zlim3d(30,160)

plt.show()