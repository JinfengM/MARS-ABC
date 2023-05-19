# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 08:51:33 2022

@author: Dell
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import chaospy
from matplotlib.pyplot import MultipleLocator
from datetime import datetime
timestart=datetime.now()
filename='6561.csv'
input_csv=pd.read_csv(filename,header=None)
paras_list=[i for i in range(8)]
#print(input_csv.iloc[:,paras_list])

#参数列表
pd_paras_list=input_csv.iloc[:,paras_list]

sim1_list=[i+9 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
#第一个站点模拟值
pd_sim1_list=input_csv.iloc[:,sim1_list]

obs1_list=[i+70 for i in range(60)]
pd_obs1_list=input_csv.iloc[:1,obs1_list]

#第二个站点模拟值
sim2_list=[i+131 for i in range(60)]
#print(input_csv.iloc[:2,sim2_list])
pd_sim2_list=input_csv.iloc[:,sim2_list]

obs2_list=[i+192 for i in range(60)]
pd_obs2_list=input_csv.iloc[:1,obs2_list]

#第三个站点模拟值
sim3_list=[i+253 for i in range(60)]
#print(input_csv.iloc[:2,sim3_list])
pd_sim3_list=input_csv.iloc[:,sim3_list]

obs3_list=[i+314 for i in range(60)]
pd_obs3_list=input_csv.iloc[:1,obs3_list]

#列表转tuple
lst_tuple_sim1=[]
list_sim1=pd_sim1_list.values.tolist()
for item in list_sim1:
    a1=tuple(item)
    lst_tuple_sim1.append(a1)
    
lst_tuple_sim2=[]
list_sim2=pd_sim2_list.values.tolist()
for item in list_sim2:
    a2=tuple(item)
    lst_tuple_sim2.append(a2)

lst_tuple_sim3=[]
list_sim3=pd_sim3_list.values.tolist()
for item in list_sim3:
    a3=tuple(item)
    lst_tuple_sim3.append(a3)

mergelstlst=[]
mergelsttup=[]

for i in range(len(lst_tuple_sim1)):
    temp=[]
    temp.append(list_sim1[i])
    temp.append(list_sim2[i])
    temp.append(list_sim3[i])
    mergelstlst.append(temp)
    
    temp2=[]
    temp2.append(lst_tuple_sim1[i])
    temp2.append(lst_tuple_sim2[i])
    temp2.append(lst_tuple_sim3[i])
    mergelsttup.append(temp2)

#MARS的参数归一化
#输入参数标准化
def MaxMinNormalization(x,Min,Max):
    x = (x-Min)/(Max-Min)
    return x

def Conversion(x,Min,Max):
    x = Min+x*(Max-Min)
    return x


def lmd1(x):
    return MaxMinNormalization(x,0,100)
def lmd2(x):
    return MaxMinNormalization(x,35,98)
def lmd3(x):
    return MaxMinNormalization(x,0,0.3)
def lmd4(x):
    return MaxMinNormalization(x,5,130)
def lmd5(x):
    return MaxMinNormalization(x,0,1)
def lmd6(x):
    return MaxMinNormalization(x,0,1)
def lmd7(x):
    return MaxMinNormalization(x,0,2000)
def lmd8(x):
    return MaxMinNormalization(x,0.9,2.5)

def rlmd1(x):
    return Conversion(x,0,100)
def rlmd2(x):
    return Conversion(x,35,98)
def rlmd3(x):
    return Conversion(x,0,0.3)
def rlmd4(x):
    return Conversion(x,5,130)
def rlmd5(x):
    return Conversion(x,0,1)
def rlmd6(x):
    return Conversion(x,0,1)
def rlmd7(x):
    return Conversion(x,0,2000)
def rlmd8(x):
    return Conversion(x,0.9,2.5)

l1=list(map(lmd1,pd_paras_list.iloc[:,0]))
l2=list(map(lmd2,pd_paras_list.iloc[:,1]))
l3=list(map(lmd3,pd_paras_list.iloc[:,2]))
l4=list(map(lmd4,pd_paras_list.iloc[:,3]))
l5=list(map(lmd5,pd_paras_list.iloc[:,4]))
l6=list(map(lmd6,pd_paras_list.iloc[:,5]))
l7=list(map(lmd7,pd_paras_list.iloc[:,6]))
l8=list(map(lmd8,pd_paras_list.iloc[:,7]))
l=np.vstack((l1,l2,l3,l4,l5,l6,l7,l8))
pd_paras_list=l.T
pd_paras_list=pd.DataFrame(pd_paras_list)


print('timecost')
a=pd_paras_list.iloc[:,:]

X=a.apply(lambda x:tuple(x), axis=1).values.tolist()
#选择不同站点‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’1
#同一套参数输入，三个时间序列输出
y1=lst_tuple_sim1
y2=lst_tuple_sim2
y3=lst_tuple_sim3

#y=mergelsttup
from pyearth import Earth
#三个MARS模型
model1 = Earth()
model2 = Earth()
model3 = Earth()

#X已经标准化到（0,1）范围
model1.fit(X,y1)
model2.fit(X,y2)
model3.fit(X,y3)

#(1)ABC过程

def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

def objectiveFunctionNSE(x):
    #print(x)
    temp=[]
    temp.append(tuple(x))
    x=temp
    
    sim1=model1.predict(x)
    obs1=list(np.array(pd_obs1_list).flat)
    obj1=(1-nse(np.array(sim1),np.array(obs1)))
    
    sim2=model2.predict(x)
    obs2=list(np.array(pd_obs2_list).flat)
    obj2=(1-nse(np.array(sim2),np.array(obs2)))
        
    sim3=model3.predict(x)
    obs3=list(np.array(pd_obs3_list).flat)
    obj3=(1-nse(np.array(sim3),np.array(obs3)))
    
    #print(obs)
    #print(nse(sim,obs))
    return[obj1,obj2,obj3]

import abcpmc
import matplotlib.pyplot as plt
#x为返回的结果列表，y为观测值列表
def dist(x, y):
    nse1=1-nse(x[0],y[0])
    nse2=1-nse(x[1],y[1])
    nse3=1-nse(x[2],y[2])
    return [nse1,nse2,nse3]

#一个参数三个结果
#abcpmc.TophatPrior(min, max)
def postfn(theta):
    paras=[theta]
    result1=model1.predict(paras).flatten()
    result2=model2.predict(paras).flatten()
    result3=model3.predict(paras).flatten()
    return result1,result2,result3

T=50
alpha=85
eps_start=[1,1,1]
eps = abcpmc.ConstEps(T, eps_start)
data=pd_obs1_list.values.tolist()+pd_obs2_list.values.tolist()+pd_obs3_list.values.tolist()
prior = abcpmc.TophatPrior([0]*8, [1]*8)
sampler = abcpmc.Sampler(500, Y=data, postfn=postfn, dist=dist)
sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal

pools = []
for pool in sampler.sample(prior, eps):
    eps_str = ", ".join(["{0:>.4f}".format(e) for e in pool.eps])

    print("T: {0}, eps: [{1}], ratio: {2:>.4f}".format(pool.t, eps_str, pool.ratio))
    for i, (mean, std) in enumerate(zip(*abcpmc.weighted_avg_and_std(pool.thetas, pool.ws, axis=0))):
        print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
    print('dists:')
    print(pool.dists)
    eps.eps = np.percentile(pool.dists, alpha,axis=0) # reduce eps value
    pools.append(pool)
    #if pool.ratio <0.005:
    #    break
sampler.close()
#ABC绘制图片

timeafter=datetime.now()
timecost=timeafter-timestart
print('timecost')
print(timecost)
#参收收敛图
labels=['CANMX','CN2','CH_N2','CH_K2','ALPHA_BNK','SOL_AWC','SOL_K','SOL_BD']
T = len(pools)
plt.figure(figsize=(12,6), dpi=80)
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.tight_layout()
    moments = np.array([abcpmc.weighted_avg_and_std(pool.thetas[:,i], pool.ws, axis=0) for pool in pools])
    plt.errorbar(range(T), moments[:, 0], moments[:, 1], label=labels[i],ecolor='r',elinewidth=0.5,capsize=1,capthick=0.5)
    plt.ylim(0,1)
    plt.ylabel(labels[i])
    plt.xlabel('Iterations')
plt.savefig('marsabcfigures/8paras-convergence.png',dpi=300)

#距离函数收敛图：pool为index(T-1)的对象
distances = np.array([pool.dists for pool in pools])
fig, ax = plt.subplots()
ax.errorbar(np.arange(len(distances)), np.mean(distances, axis=1)[:, 0], np.std(distances, axis=1)[:, 0], label='S1',ecolor='r',elinewidth=0.5,capsize=1,capthick=0.5)
ax.errorbar(np.arange(len(distances)), np.mean(distances, axis=1)[:, 1], np.std(distances, axis=1)[:, 1], label='S2',ecolor='r',elinewidth=0.5,capsize=1,capthick=0.5)
ax.errorbar(np.arange(len(distances)), np.mean(distances, axis=1)[:, 2], np.std(distances, axis=1)[:, 2], label='S3',ecolor='r',elinewidth=0.5,capsize=1,capthick=0.5)
ax.legend()
ax.set_xlabel("Iteration")
ax.set_title("Distances")
plt.savefig('marsabcfigures/distance',dpi=300)

#阈值收敛
fig,ax = plt.subplots()
eps_values = np.array([pool.eps for pool in pools])
ax.plot(eps_values[:, 0], label='S1')
ax.plot(eps_values[:, 1], label='S2')
ax.plot(eps_values[:, 2], label='S3')
ax.set_xlabel("Iteration")
#ax.set_ylabel(r"$\epsilon$", fontsize=15)
ax.legend(loc="best")
ax.set_title("Thresholds")
plt.savefig('marsabcfigures/Thresholds',dpi=300)

#使用corner绘制
import corner
samples = np.vstack([pool.thetas for pool in pools])
data=samples
figure = corner.corner(
    data,
    labels=['CANMX','CN2','CH_N2','CH_K2','ALPHA_BNK','SOL_AWC','SOL_K','SOL_BD'],
    range=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
figure.gca().annotate(
    "",
    xy=(1.0, 1.0),
    xycoords="figure fraction",
    xytext=(-20, -10),
    textcoords="offset points",
    ha="right",
    va="top",
)
figure.savefig("SWAT-MARS-3-ABC-CORNER.png", dpi=300)

#取出最后一个pool
poolEx=pools[T-1]
pd_csv=pd.DataFrame(poolEx[3])
pd_csv[0]=pd_csv[0].apply(rlmd1)
pd_csv[1]=pd_csv[1].apply(rlmd2)
pd_csv[2]=pd_csv[2].apply(rlmd3)
pd_csv[3]=pd_csv[3].apply(rlmd4)
pd_csv[4]=pd_csv[4].apply(rlmd5)
pd_csv[5]=pd_csv[5].apply(rlmd6)

pd_csv[6]=pd_csv[6].apply(lambda x: x if x>0 else 0)
pd_csv[6]=pd_csv[6].apply(rlmd7)

pd_csv[7]=pd_csv[7].apply(rlmd8)

pd_csv.to_csv('abc.csv',header=False,index=False)
#输出500个sample的参数值，写入
#多目标优化合成到一个里面来

'''
#（2）MOEAs过程
#2022-12-29屏蔽，用ABC
model=model1
#Print the model
print(model.trace())
print(model.summary())

#这里已经得到三个函数
from platypus import Problem
from platypus.algorithms import NSGAII
from platypus.types import Real
from datetime import datetime
def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

def objectiveFunctionNSE(x):
    #print(x)
    temp=[]
    temp.append(tuple(x))
    x=temp
    
    sim1=model1.predict(x)
    obs1=list(np.array(pd_obs1_list).flat)
    obj1=(1-nse(np.array(sim1),np.array(obs1)))
    
    sim2=model2.predict(x)
    obs2=list(np.array(pd_obs2_list).flat)
    obj2=(1-nse(np.array(sim2),np.array(obs2)))
    
    sim3=model3.predict(x)
    obs3=list(np.array(pd_obs3_list).flat)
    obj3=(1-nse(np.array(sim3),np.array(obs3)))
    
    #print(obs)
    #print(nse(sim,obs))
    return[obj1,obj2,obj3]

from platypus import *
print('......MOEAs过程......')
#timebefore=datetime.datetime.now()
problem = Problem(8, 3)
problem.types[:] = [Real(0,100),Real(35,98),Real(0,0.3),Real(5,130),Real(0,1.0),Real(0,1.0),Real(0,2000),Real(0.9,2.5)]
problem.function = objectiveFunctionNSE
problem.directions[:] = Problem.MINIMIZE

names=['NSGAII','NSGAIII','CMAES','GDE3','IBEA','MOEAD','OMOPSO','SMPSO','SPEA2','EpsMOEA']
population_size=1000
algorithms=[]

algorithm = NSGAII(problem,population_size = population_size)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = NSGAIII(problem,population_size = population_size,divisions_outer=12)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = CMAES(problem,population_size = population_size,epsilons=[0.005])
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = GDE3(problem,population_size = population_size)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = IBEA(problem,population_size = population_size)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = MOEAD(problem,population_size = population_size,weight_generator=normal_boundary_weights,divisions_outer=12)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = OMOPSO(problem,population_size = population_size,epsilons=[0.005])
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = SMPSO(problem,population_size = population_size)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = SPEA2(problem,population_size = population_size)
algorithm.run(10000)
algorithms.append(algorithm)

algorithm = EpsMOEA(problem,population_size = population_size,epsilons=[0.005])
algorithm.run(10000)
algorithms.append(algorithm)

#timeafter=datetime.now()
#timecostNSAG=timeafter-timebefore
#print('timecostNSAG: '+str(timecostNSAG))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

for i in range(10):
    algorithm=algorithms[i]
    fig=plt.figure()
    plt.rc('font',family='Times New Roman',size=14)
    ax=fig.add_subplot(111,projection='3d')
    
    ax.scatter([1-s.objectives[0] for s in algorithm.result],
                [1-s.objectives[1] for s in algorithm.result],
                [1-s.objectives[2] for s in algorithm.result],s=5)

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_xlim([0.8,0.95])
    ax.set_ylim([0.65,0.8])
    ax.set_zlim([0.8,0.95])
    ax.view_init(elev=30.0, azim=15.0)
    ax.locator_params(nbins=4)
    plt.savefig('marsabcfigures/swat-mars-1000-'+names[i]+'.png', dpi=300)
    plt.show()
    
    #这里导出NSGA的pareto前沿
    
    savefilename=open('marsabcfigures/SWAT-MARS-'+names[i]+'.csv','w')
    for sample in algorithm.result:
        for i in range(len(sample.variables)):
            savefilename.writelines(str(sample.variables[i]))
            savefilename.writelines(',')
            nsesum=0
        for j in range(len(sample.objectives)-1):
            nsesum+=sample.objectives[j]
            savefilename.writelines(str(sample.objectives[j]))
            savefilename.writelines(',')
        nsesum+=sample.objectives[len(sample.objectives)-1]
        savefilename.writelines(str(sample.objectives[len(sample.objectives)-1]))
        savefilename.writelines(',')
        savefilename.writelines(str(nsesum))
        savefilename.writelines('\n')
    savefilename.close()
'''
