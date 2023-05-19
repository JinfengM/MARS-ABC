# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:23:29 2022

@author: Dell
"""

import numpy as np
import pandas as pd
import glob
import re
'''
csv_list = glob.glob('*.csv')
print('发现%s个CSV文件'% len(csv_list))
print('对文件进行处理中')
for i in csv_list:
    fr = open(i,'r',encoding='utf-8').read()
    with open('合并之后的文件.csv','a',encoding='utf-8') as f:
        f.write(fr)
print('所有文件合并完成！')
'''

filename='合并之后的文件.csv'
input_csv=pd.read_csv(filename,header=None)

def MaxMinNormalization(x,Min,Max):
    x = (x-Min)/(Max-Min)
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

l1=list(map(lmd1,input_csv.iloc[:,0]))
l2=list(map(lmd2,input_csv.iloc[:,1]))
l3=list(map(lmd3,input_csv.iloc[:,2]))
l4=list(map(lmd4,input_csv.iloc[:,3]))
l5=list(map(lmd5,input_csv.iloc[:,4]))
l6=list(map(lmd6,input_csv.iloc[:,5]))
l7=list(map(lmd7,input_csv.iloc[:,6]))
l8=list(map(lmd8,input_csv.iloc[:,7]))
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.hist(l1,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('CANMX')
ax.set_ylabel('Frequency')
fig.savefig('fig1.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l2,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('CN2')
ax.set_ylabel('Frequency')
fig.savefig('fig2.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l3,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('CH_N2')
ax.set_ylabel('Frequency')
fig.savefig('fig3.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l4,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('CH_K2')
ax.set_ylabel('Frequency')
fig.savefig('fig4.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l5,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('ALPHA_BNK')
ax.set_ylabel('Frequency')
fig.savefig('fig5.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l6,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('SOL_AWC')
ax.set_ylabel('Frequency')
fig.savefig('fig6.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l7,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('SOL_K')
ax.set_ylabel('Frequency')
fig.savefig('fig7.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l8,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('SOL_BD')
ax.set_ylabel('Frequency')
fig.savefig('fig8.png',dpi=300)

#绘制corner图像
lall=[l1,l2,l3,l4,l5,l6,l7,l8]
mydata=[list(x) for x in zip(*lall)]
mydata=np.array(mydata)
import corner
figure = corner.corner(
    mydata,
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
figure.savefig("demo.png", dpi=300)
'''
for i in range(8):
    fig, ax = plt.subplots()
    a=input_csv.iloc[:,i]
    plt.hist(a,25, density=False, facecolor='g', alpha=0.75)
    
    fig.savefig('fig'+str(i)+'.png',dpi=300)
    
'''