import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sigmod(x):
    return 1/(1+np.exp(-x))
def de_sigmod(x):
    return (x-x**2)
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def de_tanh(x):
    return (1-x**2)


#参数设置
samnum = 800
hiddenunitnum = 8
indim = 4
outdim = 1
maxepochs = 100
errorfinal = 0.65*10**(-3)
learnrate = 0.001

df = pd.read_csv("2.csv")  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
df.columns = ["x", "y", "high", "use", "As"]
longtitude = df["x"] # 抽取前四列作为训练数据的各属性值
longtitude = np.array(longtitude)
latitude = df["y"]
latitude = np.array(latitude)
elevation = df["high"]
elevation =np.array(elevation)
functional = df["use"]
functional = np.array(functional)
ag=df["As"]
ag=np.array(ag)

samplein = np.mat([longtitude,latitude,elevation,functional])  #4*1000
sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()#对应最大值最小值
sampleout = np.mat([ag])
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()#对应最大值最小值
#归一化方法1
#sampleinnorm = (2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])-1).transpose()
#sampleoutnorm = (2*(np.array(sampleout.T)-sampleoutminmax.transpose()[0])/(sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])-1).transpose()
#归一化方法2，最大最小
sampleinnorm = ((np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])).transpose()
sampleoutnorm = ((np.array(sampleout.T)-sampleoutminmax.transpose()[0])/(sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])).transpose()


sampleinmax = np.array([sampleinnorm.max(axis=1).T.tolist()]).transpose()
sampleinmin = np.array([sampleinnorm.min(axis=1).T.tolist()]).transpose()
sampleoutmax = np.array([sampleoutnorm.max(axis=1).T.tolist()]).transpose()
sampleoutmin = np.array([sampleoutnorm.min(axis=1).T.tolist()]).transpose()
noise = 0.03*np.random.rand(sampleoutnorm.shape[0],sampleoutnorm.shape[1])
sampleoutnorm += noise


x = sampleinnorm.transpose()
estimator=KMeans(n_clusters=8,max_iter=10000)
estimator.fit(x)
w1 = estimator.cluster_centers_


#b为8*1
b1 = np.mat(np.zeros((hiddenunitnum,outdim)))
for i in range(hiddenunitnum):
    cmax = 0
    for j in range(hiddenunitnum):
        temp_dist=np.sqrt(np.sum(np.square(w1[i,:]-w1[j,:])))
        if cmax<temp_dist:
            cmax=temp_dist
    b1[i] = cmax/np.sqrt(2*hiddenunitnum)


#生成方法1
scale = np.sqrt(3/((indim+outdim)*0.5))
w2 = np.random.uniform(low=-scale,high=scale,size=[hiddenunitnum,outdim])
b2 = np.random.uniform(low=-scale, high=scale, size=[outdim,1])


inputin=np.mat(sampleinnorm.T)
w1=np.mat(w1)
b1=np.mat(b1)
w2=np.mat(w2)
b2=np.mat(b2)


errhistory = np.mat(np.zeros((1,maxepochs)))
#开始训练
for i in range(maxepochs):
    hidden_out = np.mat(np.zeros((samnum,hiddenunitnum)))
#前向计算：
    for a in range(samnum):
        for j in range(hiddenunitnum):
            d=(inputin[a, :] - w1[j, :]) * (inputin[a, :] - w1[j, :]).T
            c=2 * b1[j, :] * b1[j, :]
            hidden_out[a, j] = np.exp((-1.0 )* (d/c))
    # print(hidden_out.shape,"33333333333")
    output = tanh(hidden_out * w2 + b2)
    # output = sigmod(hidden_out * w2)
    # 计算误差
    out_real = np.mat(sampleoutnorm.transpose())
    err = out_real - output
    loss = np.sum(np.square(err))
    if loss < errorfinal:
        break
    errhistory[:,i] = loss
#反向计算
    output=np.array(output.T)
    belta=de_tanh(output).transpose()

    dw1now = np.zeros((8,4))
    db1now = np.zeros((8,1))
    dw2now = np.zeros((8,1))
    db2now = np.zeros((1,1))
    for j in range(hiddenunitnum):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        for a in range(samnum):
            #1*4
            sum1 +=err[a,:] * belta[a,:] * hidden_out[a,j] * (inputin[a,:]-w1[j,:])
            #1*1
            sum2 +=err[a,:] * belta[a,:] * hidden_out[a,j] * (inputin[a,:]-w1[j,:])*(inputin[a,:]-w1[j,:]).T
            #1*1
            sum3 +=err[a,:] * belta[a,:] * hidden_out[a,j]
            sum4 +=err[a,:] * belta[a,:]
        dw1now[j,:]=(w2[j,:]/(b1[j,:]*b1[j,:])) * sum1
        db1now[j,:] =(w2[j,:]/(b1[j,:]*b1[j,:]*b1[j,:])) * sum2
        dw2now[j,:] =sum3
        db2now = sum4

    w1 += learnrate * dw1now
    b1 += learnrate * db1now
    w2 += learnrate * dw2now
    b2 += learnrate * db2now
    print("the iteration is:",i+1,",the loss is:",loss)

print('更新的权重w1:',w1)
print('更新的偏置b1:',b1)
print('更新的权重w2:',w2)
print('更新的偏置b2:',b2)
print("The loss after iteration is ：",loss)

np.save("w1.npy",w1)
np.save("b1.npy",b1)
np.save("w2.npy",w2)
np.save("b2.npy",b2)
np.save("rbf_err.npy",errhistory)

