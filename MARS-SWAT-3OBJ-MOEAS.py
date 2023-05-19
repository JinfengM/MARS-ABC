# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import chaospy
from matplotlib.pyplot import MultipleLocator

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

model1.fit(X,y1)
model2.fit(X,y2)
model3.fit(X,y3)

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
#timebefore=datetime.datetime.now()
problem = Problem(8, 3)
problem.types[:] = [Real(0,100),Real(35,98),Real(0,0.3),Real(5,130),Real(0,1.0),Real(0,1.0),Real(0,2000),Real(0.9,2.5)]
problem.function = objectiveFunctionNSE
problem.directions[:] = Problem.MINIMIZE

#algorithm = NSGAII(problem,population_size = 100)
#algorithm = NSGAIII(problem,population_size = 100,divisions_outer=12)
algorithm = CMAES(problem,population_size = 100,epsilons=[0.005])
#algorithm = GDE3(problem,population_size = 100)
#algorithm = IBEA(problem,population_size = 100)
#algorithm = MOEAD(problem,population_size = 100,weight_generator=normal_boundary_weights,divisions_outer=12)
#algorithm = OMOPSO(problem,population_size = 100,epsilons=[0.005])
#algorithm = SMPSO(problem,population_size = 100)
#algorithm = SPEA2(problem,population_size = 100)
#algorithm = EpsMOEA(problem,population_size = 100,epsilons=[0.005])
algorithm.run(10000)

#timeafter=datetime.now()
#timecostNSAG=timeafter-timebefore
#print('timecostNSAG: '+str(timecostNSAG))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
plt.savefig('swat-mars-1000-CMAES.png', dpi=300)
plt.show()

#这里导出NSGA的pareto前沿

savefilename=open('./SWAT-MARS-CMAES.csv','w')
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


