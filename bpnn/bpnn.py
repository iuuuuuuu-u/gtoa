import numpy as np
import pandas as pd

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def de_tanh(x):
    return (1-x**2)

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
As=df["As"]
As=np.array(As)
samplein = np.mat([longtitude,latitude,elevation,functional])

# df = pd.read_csv("2.csv")  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
# df.columns = ["Cr", "Cu", "Ni", "Pb", "Zn","As"]
# Cr = df["Cr"] # 抽取前四列作为训练数据的各属性值
# Cr = np.array(Cr)
# Cu = df["Cu"]
# Cu = np.array(Cu)
# Ni= df["Ni"]
# Ni =np.array(Ni)
# Pb = df["Pb"]
# Pb = np.array(Pb)
# Zn = df["Zn"]
# Zn = np.array(Zn)
# As=df["As"]
# As=np.array(As)
# samplein = np.mat([Cr,Cu,Ni,Pb,Zn])

sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()
sampleout = np.mat([As])
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()
sampleinnorm = (2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])-1).transpose()
sampleoutnorm = (2*(np.array(sampleout.T)-sampleoutminmax.transpose()[0])/(sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])-1).transpose()
noise = 0.03*np.random.rand(sampleoutnorm.shape[0],sampleoutnorm.shape[1])
sampleoutnorm += noise


maxepochs = 5000
learnrate = 0.001
errorfinal = 0.65*10**(-3)
samnum = 800
indim = 4
outdim = 1
hiddenunitnum = 8

scale = np.sqrt(3/((indim+outdim)*0.5))  #最大值最小值范围为-1.44~1.44
w1 = np.random.uniform(low=-scale, high=scale, size=[hiddenunitnum,indim])
b1 = np.random.uniform(low=-scale, high=scale, size=[hiddenunitnum,1])
w2 = np.random.uniform(low=-scale, high=scale, size=[outdim,hiddenunitnum])
b2 = np.random.uniform(low=-scale, high=scale, size=[outdim,1])


errhistory = np.mat(np.zeros((1,maxepochs)))

for i in range(maxepochs):
    print("The iteration is : ", i)
    hiddenout = tanh((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
    networkout = tanh((np.dot(w2,hiddenout).transpose()+b2.transpose())).transpose()
    err = sampleoutnorm - networkout
    loss = np.sum(err**2)/2
    print("the loss is :",loss)
    errhistory[:,i] = loss
    if loss < errorfinal:
        break
    delta2 = err*de_tanh(networkout)
    delta1 = np.dot(w2.transpose(),delta2)*de_tanh(hiddenout)
    dw2 = np.dot(delta2,hiddenout.transpose())
    db2 = np.dot(delta2,np.ones((samnum,1)))
    dw1 = np.dot(delta1,sampleinnorm.transpose())
    db1 = np.dot(delta1,np.ones((samnum,1)))
    dw2 = dw2/samnum
    dw1 = dw1/samnum
    db1 = db1/samnum
    db2 = db2/samnum

    w2 += learnrate*dw2
    b2 += learnrate*db2
    w1 += learnrate*dw1
    b1 += learnrate*db1
print('更新的权重w1:',w1)
print('更新的偏置b1:',b1)
print('更新的权重w2:',w2)
print('更新的偏置b2:',b2)
print("The loss after iteration is ：",loss)

np.save("w1.npy",w1)
np.save("b1.npy",b1)
np.save("w2.npy",w2)
np.save("b2.npy",b2)
np.save("bp_err.npy",errhistory)