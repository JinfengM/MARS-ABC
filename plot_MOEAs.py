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

filename='合并之后的文件多目标优化-结果2.csv'
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

l1=input_csv.iloc[:,9]
l2=input_csv.iloc[:,10]
l3=input_csv.iloc[:,11]
l4=input_csv.iloc[:,12]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.hist(l1,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('S1')
ax.set_ylabel('Frequency')
ax.set_ylim([0,250])
fig.savefig('moea1.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l2,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('S2')
ax.set_ylabel('Frequency')
ax.set_ylim([0,250])
fig.savefig('moea2.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l3,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('S3')
ax.set_ylabel('Frequency')
ax.set_ylim([0,250])
fig.savefig('moea3.png',dpi=300)

fig, ax = plt.subplots()
plt.hist(l4,25, density=False, facecolor='g', alpha=0.75,range=[0,1])
ax.set_xlabel('Average')
ax.set_ylabel('Frequency')
ax.set_ylim([0,250])
fig.savefig('moea4.png',dpi=300)


