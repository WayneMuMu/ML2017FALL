import numpy as np
import pandas as pd
import sys

# Read Data 
x_data = pd.read_csv(sys.argv[1], sep = ',' ,encoding = 'UTF-8')
y_data = pd.read_csv(sys.argv[2], sep = ',' ,encoding = 'UTF-8')

x_data = x_data.astype(float)
y_data = y_data.astype(float)
num_win = int(np.sum(y_data))
num_total = len(y_data['label'])
num_lose = num_total - num_win
p_win = np.mean(y_data['label'])
p_lose = 1- p_win
x_data = pd.concat((x_data,y_data),axis =1)
x_data = x_data.sort_values(by = 'label' , ascending = 0)
x_data = x_data.as_matrix()
_max = np.max(x_data, axis = 0)
x_data /= _max 

# Select Feature
age = [0]
fnlwgt = [1]
sex = [2]
capital_gain = [3]
capital_loss = [4]
hours_per_week = [5] 
workclass = [i for i in range(6,15)]
education_num = [i for i in range(15,22)]
education = [i for i in range(22,31)]
marital_status = [i for i in range(31,38)]
occupation = [i for i in range(38,53)]
relationship = [i for i in range(53,59)]
race = [i for i in range(59,64)]
country = [i for i in range(64,106)]

#sel_array = np.array(race)
sel_array = np.concatenate((age,sex,capital_gain,capital_loss,workclass,education_num,education,marital_status,occupation,race,country))
x_win = x_data[0:num_win,sel_array]
x_lose = x_data[num_win:num_total,sel_array]
x_win = np.concatenate((x_win,x_win **2), axis =1)
x_lose = np.concatenate((x_lose,x_lose **2), axis =1)
x_win = np.concatenate((x_win,np.ones((num_win,1))), axis = 1)
x_lose = np.concatenate((x_lose,np.ones((num_lose,1))), axis = 1)
"""
# Gradient Descent
lrn_rate = 1
iteration = 10000
w = np.zeros((1,2*len(sel_array)+1))
lrn =  np.zeros((1,2*len(sel_array)+1))

for i in range(iteration):
	f_win = 1/(1+np.exp(-np.sum(x_win*w, axis = 1)))
	#print(f_win)
	f_lose = 1/(1+np.exp(-np.sum(x_lose*w, axis = 1)))
	#print(f_lose)
	if i%100 == 0:
		Loss = (np.sum(-np.log(f_win))+ np.sum(-np.log(1-f_lose)))
		print(Loss)

	err_win = np.sum(-x_win*((1 - f_win).reshape(-1,1)), axis = 0)
	err_lose = np.sum(-x_lose*((-f_lose).reshape(-1,1)), axis = 0)
	
	grad_w = err_win + err_lose
	lrn = lrn + grad_w ** 2
	w -= lrn_rate/np.sqrt(lrn)*grad_w

# Error rate
err_train_win = np.sum((f_win<=0.5).astype(int))
err_train_lose = np.sum((f_lose>0.5).astype(int))
err_train = err_train_win + err_train_lose
print("Count: %d | Error Rate: %f" %(err_train,err_train/num_total))

# Save Model
np.save('logistic_2nd_w.npy',w)
np.save('logistic_2nd_max.npy',_max)
"""
############################################################################

# Read Model
w = np.load('logistic_2nd_w.npy')
_max = np.load('logistic_2nd_max.npy')

# Read test_data
test_data = pd.read_csv(sys.argv[3], encoding = 'UTF-8')
test_data = test_data.astype(float).as_matrix()
test_data = np.concatenate((test_data,np.ones((test_data.shape[0],1))),axis = 1)
test_data /= _max 
test_data = test_data[:,sel_array]
test_data = np.concatenate((test_data, test_data **2), axis =1)
test_data = np.concatenate((test_data,np.ones((test_data.shape[0],1))),axis = 1)
output = []
for i in range(test_data.shape[0]):
	eva = 1/(1+np.exp(-np.sum(test_data[i,:]*w)))
	if eva > 0.5:
		outcome = 1
	else:
		outcome = 0
	output.append(outcome)

result=pd.DataFrame()
result['id'] = [str(i) for i in range(1,test_data.shape[0]+1)]
result['label'] = pd.Series(output, index = result.index)
result.to_csv(sys.argv[4],index=False)








