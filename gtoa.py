import numpy as np
import random

Tmax=0;  #最大评估次数
Tcur=0;  #当前评估次数
N=10;   #种群大小
u=20;   #决策变量上界
l=-20;  #决策变量下下界
D=1;    #维度大小
# Y=x^2-5;    #适应度函数


#小初始化种群，随机生成种群
a=0
X=np.zeros((1,10));
for j in range(N) :
    i=l+(u-l)* random.uniform(0,1)
    X[0][a]=i;
    a=a+1;

#获取适应度，拿到当前最优解
Y=np.subtract(np.multiply(X,X),5)
Tcur=Tcur+N

#当前最优解
G=Y[0][Y.argmin()]
print(Y)

print(G)

#设置终止条件，当到达迭代次数时，跳出循环，输出最优解
flag=1;
while flag:
    if Tcur>=Tmax:
        print(G)
        flag=0;
    else :
        pass