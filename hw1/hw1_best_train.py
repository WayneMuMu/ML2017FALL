import numpy as np
import pandas as pd
import sys

NR_Replace = '-5'
dim = 18
choose = 9
reg = 5
days = 240

df = pd.read_csv(sys.argv[1], encoding='big5')
df.replace(to_replace='NR',value=NR_Replace,inplace=True)
del(df['日期'],df['測站'],df['測項'])
df = df.astype(float)
data = df.as_matrix()
data = data.T.reshape(240*24,18)

del_array = np.array([0,1,2,3,4,5,6,11,12,13,15,17])
data = np.delete(data,del_array,1)
dim -= len(del_array)
forwdim = sum((del_array<9).astype(int))
data = data.reshape(24,240*dim).T

w_list = list()
y_list = list()

for k in range(days):
	for m in range(15):
		x = data[k*dim:(k+1)*dim,m+9-choose:m+9].T.reshape(1,dim*choose)
		w_list.append(x)
		y_list.append(data[k*dim+9-forwdim,m+9])

w2_array = np.asarray(w_list).reshape(days*15,dim*choose) **2
w_array = np.asarray(w_list).reshape(days*15,dim*choose)
b_array = np.ones((days*15,1))
y_array = np.asarray(y_list).reshape(days*15,1)

array = np.concatenate((w2_array,w_array,b_array), axis = 1)
array_pinv = np.linalg.pinv(array.T.dot(array)+ reg*np.identity(array.shape[1]))

output = (array_pinv.dot(array.T)).dot(y_array)

np.save('hw1_best_w.npy',output)

#print(output)
#print(output.shape)

#############################################################################################


output = np.load('hw1_best_w.npy')
df_out = pd.read_csv(sys.argv[2], encoding='big5',header=None)
df_out.replace(to_replace='NR',value=NR_Replace,inplace=True)
del(df_out[0],df_out[1])
df_out.columns =[i for i in range(9)]
df_out = df_out.astype(float)
out_data = df_out.as_matrix().T.reshape(240*9,18)
out_data = np.delete(out_data,del_array,1)
out_data = out_data.reshape(9,240*dim).T

output_list = list()
for k in range(240):
	x2 = out_data[k*dim:(k+1)*dim,9-choose:9].T.reshape(dim*choose,1) **2
	x1 = out_data[k*dim:(k+1)*dim,9-choose:9].T.reshape(dim*choose,1)
	x0 = np.ones((1,1)).reshape(1,1)
	x = np.concatenate((x2,x1,x0), axis = 0)
	y = max(np.sum(output*x),0)
	output_list.append(y)

result=pd.DataFrame()
result['id'] = ['id_'+str(i) for i in range(240)]
result['value'] = pd.Series(output_list, index = result.index)
result.to_csv('second_pinv.csv',index=False)

print(result)
