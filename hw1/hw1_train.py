import numpy as np
import pandas as pd
import sys

# Initialize
NR_Replace = '0'
dim = 18  # Be awarw of DIY or SVD 
choose = 9 # <= 9
forwdim = 0
days =240

# Read Data
df = pd.read_csv(sys.argv[1], encoding='big5')
df.replace(to_replace='NR',value=NR_Replace,inplace=True)
del(df['日期'],df['測站'],df['測項'])
df = df.astype(float)
data = df.as_matrix()
data = data.T.reshape(240*24,18)

# DIY or not

del_array = np.array([0,1,2,3,4,5,6,11,12,13,14,15,16,17])
data = np.delete(data,del_array,1)
dim -= len(del_array)
forwdim = sum((del_array<9).astype(int))
#"""

# Read y before SVD
mean = np.mean(data, axis = 0)
std  = np.std( data, axis = 0)
data = (data - mean)/std
data = data.reshape(24,240*dim).T
y_list = list()
for k in range(days):
	for m in range(15):
		y = data[k*dim+9-forwdim,m+9]
		y_list.append(y)

# SVD or not , dim not yet
"""
data = data.T.reshape(240*24,dim)
sigma = data.T.dot(data)/(240*24)
dim = 18
U,s,V = np.linalg.svd(sigma, full_matrices = True)
U_reduce = U[:,0:dim]
data = data.dot(U_reduce).reshape(24,240*dim).T
#"""

# Read x Data after SVD
x_list = list()
for k in range(days):
	for m in range(15):
		x = data[k*dim:(k+1)*dim,m+9-choose:m+9].T.reshape(1,dim*choose)
		x_list.append(x)

# x,y data to array
x_array = np.asarray(x_list).reshape(days*15,dim*choose)
x_array = np.concatenate((x_array,x_array **2),axis =1)
y_array = np.asarray(y_list).reshape(days*15,1)

# Parameter
iteration = 10000
reg = 0.001

w =np.ones(len(x_array[0]))/10
b = 0
lrn_w = np.zeros(len(x_array[0]))
lrn_b = 0
grad_w = np.zeros(len(x_array[0]))
grad_b = 0
lrn_rate_w = 1
lrn_rate_b = 1

# Gradient Descent
for i in range(iteration):
	#grad_w = np.zeros(len(x_array[0]))
	#grad_b = 0

	wTx_h = np.sum(x_array*w, axis=1).reshape(days*15,1)
	wTx = wTx_h + b*np.ones((days*15,1))
	error = (y_array - wTx)
	temp_rms = np.sqrt(np.sum(((error*std[9-forwdim])) **2)/(days*15))
	print(temp_rms)

	grad_w = -2*(np.sum(x_array*error, axis=0)) + 2*reg*w*days*15
	grad_b = -2*np.sum(error)

	lrn_w = lrn_w + grad_w **2
	lrn_b = lrn_b + grad_b **2
	w = w -lrn_rate_w/np.sqrt(lrn_w)*grad_w
	b = b -lrn_rate_b/np.sqrt(lrn_b)*grad_b

np.save('hw1_w.npy',w)
np.save('hw1_mean.npy',mean)
np.save('hw1_std.npy',std)
np.save('hw1_b.npy',b)



