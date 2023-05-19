# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:44:32 2022

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:18:14 2021

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:49:38 2021

@author: Administrator
"""


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
print(input_csv.iloc[:2,sim1_list])
#第一个站点模拟值
pd_sim1_list=input_csv.iloc[:,sim1_list]

obs1_list=[i+70 for i in range(60)]
pd_obs1_list=input_csv.iloc[:2,obs1_list]

#第二个站点模拟值
sim2_list=[i+131 for i in range(60)]
print(input_csv.iloc[:2,sim2_list])
pd_sim2_list=input_csv.iloc[:,sim2_list]

obs2_list=[i+192 for i in range(60)]
pd_obs2_list=input_csv.iloc[:2,obs2_list]

#第三个站点模拟值
sim3_list=[i+253 for i in range(60)]
print(input_csv.iloc[:2,sim3_list])
pd_sim3_list=input_csv.iloc[:,sim3_list]

obs3_list=[i+314 for i in range(60)]
pd_obs3_list=input_csv.iloc[:2,obs3_list]

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
#所有参数
a=pd_paras_list.iloc[:,:]
X=a.apply(lambda x:tuple(x), axis=1).values.tolist()
#选择不同站点‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’                 1
y=lst_tuple_sim2
#y=mergelsttup
from pyearth import Earth
model = Earth()
model.fit(X,y)
    
#Print the model
print(model.trace())
print(model.summary())
    
#Plot the model
y_hat = model.predict(X)

x=['2010-1','2010-2','2010-3','2010-4','2010-5','2010-6','2010-7','2010-8','2010-9','2010-10','2010-11','2010-12','2011-1','2011-2','2011-3','2011-4','2011-5','2011-6','2011-7','2011-8','2011-9','2011-10','2011-11','2011-12','2012-1','2012-2','2012-3','2012-4','2012-5','2012-6','2012-7','2012-8','2012-9','2012-10','2012-11','2012-12','2013-1','2013-2','2013-3','2013-4','2013-5','2013-6','2013-7','2013-8','2013-9','2013-10','2013-11','2013-12','2014-1','2014-2','2014-3','2014-4','2014-5','2014-6','2014-7','2014-8','2014-9','2014-10','2014-11','2014-12']

#MC验证
number=10000
strtraining='Validation'
newfilename=str(number)
newinput_csv=pd.read_csv(newfilename+'.csv',header=None)

#获取参数集合
newparas_list=[i for i in range(8)]
#print(newinput_csv.iloc[:,newparas_list])
newpd_paras_list=newinput_csv.iloc[:,newparas_list]

#获取站点1模拟值
newsim1_list=[i+9 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
newpd_sim1_list=newinput_csv.iloc[:,newsim1_list]

#获取站点2模拟值
newsim2_list=[i+131 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
newpd_sim2_list=newinput_csv.iloc[:,newsim2_list]

#获取站点2模拟值
newsim3_list=[i+253 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
newpd_sim3_list=newinput_csv.iloc[:,newsim3_list]


mcobsmean=[]
mcobsvar=[]
mcvalues=[]

#配置不同站点；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；；                   2
newpd_sim_list=newpd_sim2_list

for item in range(newpd_sim_list.shape[1]):
    a=newpd_sim_list.iloc[:,[item]].mean().tolist()#均值
    b=newpd_sim_list.iloc[:,[item]].var().tolist()#方差
    mcvalues.append(newpd_sim_list.iloc[:,[item]].values)#MC模拟值
    mcobsmean.append(a)
    mcobsvar.append(b)
mcobsmean=list(np.array(mcobsmean).flat)
mcobsvar=list(np.array(mcobsvar).flat)
#(1)uniform(a,b)

#(2)uniform(a,b)->uniform(0,1)
#得到MC模拟值
#.................................................................验证数据集：              所有参数
a=newpd_paras_list.iloc[:,:]
X=a.apply(lambda x:tuple(x), axis=1).values.tolist()
#选择不同站点‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’
y_hat=model.predict(X)
newapprox_evaluations=y_hat.T
simmean=[]
simvar=[]
simvalues=[]
for item in range(newapprox_evaluations.shape[0]):
    simmean.append(newapprox_evaluations[item].mean())#均值
    simvar.append(newapprox_evaluations[item].var())#方差
    simvalues.append(newapprox_evaluations[item])#PCE近似值
    

from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np
model = linear_model.LinearRegression()
model.fit(np.array(simvar).reshape(-1,1), np.array(mcobsvar).reshape(-1,1))
a=model.intercept_
b=model.coef_
#newY=a+b*x
#生产拟合曲线
newY=a+b*simvar
newY=newY.flatten()

print(model.score(np.array(simvar).reshape(-1,1),np.array(mcobsvar).reshape(-1,1)))

import matplotlib
from matplotlib.font_manager import _rebuild
_rebuild()
#################################月均值时间序列#######################################
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(x,simmean,label='MARS',color='#0C5DA5',linewidth=0.5)
    ax.scatter(x,mcobsmean,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(12)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.legend()
    ax.set_title(strtraining)
    ax.set_ylabel('Mean monthly flow(m$^3$/s)')
    ax.set_xlabel('Date')
    ax.autoscale(tight=True)
    fig.savefig('figuremars/'+str(number)+'-MARS-mean'+newfilename+'.png', dpi=300)
    
#################################月方差时间序列#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(x,simvar,label='MARS',color='#0C5DA5',linewidth=0.5)
    #ax.plot(x,mcobsvar,label='mcobsvar',color='#FF2C00',linewidth=0.5)
    ax.scatter(x,mcobsvar,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(12)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylim([0,80])
    ax.legend()
    ax.set_title(strtraining)
    #ax.legend(title='variance方差')
    ax.set_ylabel('Flow variance(m$^3$/s)')
    #ax.autoscale(tight=True)
    fig.savefig('figuremars/'+str(number)+'-MARS-var'+newfilename+'.png', dpi=300)
    
#绘制对比图
#################################均值点对点对比#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(simmean,simmean,label='MARS',color='#0C5DA5',linewidth=0.5)
    ax.scatter(simmean,mcobsmean,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(200)
    ax.xaxis.set_major_locator(x_major_locator)
    #ax.legend()
    ax.set_title(strtraining)
    ax.set_ylabel('Mean monthly flow using MC(m$^3$/s)')
    ax.set_xlabel('Mean monthly flow using MARS(m$^3$/s)')

    #ax.autoscale(tight=True)
    fig.savefig('figuremars/'+str(number)+'-mean-compare'+newfilename+'.png', dpi=300)
    
    #################################方差点对点对比#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    #ax.plot(simvar,simvar,label='Line with unit slope',color='#0C5DA5',linewidth=0.5)
    ax.plot(simvar,newY,label='Best fit R$^2$=0.91',color='#0C5DA5',linewidth=0.5)
    ax.scatter(simvar,mcobsvar,color='#FF2C00',s=0.5,label='MC')
    x_major_locator=MultipleLocator(10000)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)
    ax.legend()
    ax.set_title(strtraining)
    ax.set_xlim([0,30000])
    ax.set_ylim([0,30000])
    ax.set_ylabel('Flow variance using MC(m$^3$/s)')
    ax.set_xlabel('Flow variance using MARS(m$^3$/s)')
    #ax.autoscale(tight=True)
    fig.savefig('figuremars/'+str(number)+'-var-compare'+newfilename+'.png', dpi=300)
    