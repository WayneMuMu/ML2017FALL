import numpy as np
import pandas as pd
import sys

NR_Replace = '0'
dim = 18  # Be awarw of DIY or SVD 
choose = 9 # <= 9
forwdim = 0
days =240

del_array = np.array([0,1,2,3,4,5,6,11,12,13,14,15,16,17])
dim -= len(del_array)
forwdim = sum((del_array<9).astype(int))

w = np.load('hw1_w.npy')
mean = np.load('hw1_mean.npy')
std = np.load('hw1_std.npy')
b = np.load('hw1_b.npy')

df_test = pd.read_csv(sys.argv[1], encoding='big5',header=None)
df_test.replace(to_replace='NR',value=NR_Replace,inplace=True)
del(df_test[0],df_test[1])
df_test.columns =[i for i in range(9)]
df_test = df_test.astype(float)
test_data = df_test.as_matrix().T.reshape(240*9,18)


test_data = np.delete(test_data,del_array,1)
test_data = (test_data - mean)/std
test_data = test_data.reshape(9,240*dim).T

output = list()
for k in range(240):
  x = test_data[k*dim:(k+1)*dim,9-choose:9].T.reshape(1,dim*choose)
  x = np.concatenate((x,x **2), axis =1)
  y = np.sum(w*x)+b
  output.append(y)

out = np.asarray(output)
out = out*std[9-forwdim]+mean[9-forwdim]

result=pd.DataFrame()
result['id'] = ['id_'+str(i) for i in range(240)]
result['value'] = pd.Series(out, index = result.index)
result.to_csv(sys.argv[2],index=False)