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
#_max = np.max(x_data, axis = 0)
#x_data /= _max 

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

sel_array = np.concatenate((age,fnlwgt,sex,capital_gain,capital_loss,hours_per_week))#,workclass,education_num,education,marital_status,occupation,relationship,race,country))
x_win = x_data[0:num_win,sel_array]
x_lose = x_data[num_win:num_total,sel_array]

# Data Proceassing
"""
_win_max = np.max(x_win, axis =0) 
win_zero = np.array([np.where(_win_max == 0.0)]).reshape(1,-1)
x_win = np.delete(x_win,win_zero,1)
print(win_zero)
_lose_max = np.max(x_lose, axis =0) 
lose_zero = np.array([np.where(_lose_max == 0.0)]).reshape(1,-1)
x_lose = np.delete(x_lose,lose_zero,1)
print(lose_zero)
"""
win_mean = np.mean(x_win, axis = 0)
x_win = (x_win - win_mean).reshape(num_win,1,-1)
win_cov = np.sum((x_win.reshape(num_win,-1,1)*x_win).reshape(num_win,-1), axis = 0).reshape(x_win.shape[2],x_win.shape[2]) / num_win

lose_mean = np.mean(x_lose, axis = 0)
x_lose = (x_lose - lose_mean).reshape(num_lose,1,-1)
lose_cov = np.sum((x_lose.reshape(num_lose,-1,1)*x_lose).reshape(num_lose,-1), axis = 0).reshape(x_lose.shape[2],x_lose.shape[2]) / num_lose

# Generative Model

def _F_WIN (x) :
	f_win = np.exp(-1/2 * (x - win_mean).dot(np.linalg.pinv(win_cov).dot((x-win_mean).T)))/ (np.abs(((2*np.pi)**len(sel_array)) * (np.linalg.det(win_cov))) ** 0.5)
	return f_win

def _F_LOSE (x) :
	f_lose = np.exp(-1/2 * (x - lose_mean).dot(np.linalg.pinv(lose_cov).dot((x-lose_mean).T)))/ (np.abs(((2*np.pi)**len(sel_array)) * (np.linalg.det(lose_cov))) ** 0.5)
	return f_lose

def _Prob (x) :
	prob = _F_WIN(x)*p_win / ( _F_WIN(x)*p_win + _F_LOSE(x)*p_lose )
	return prob

def _Classfication (x) :
	if _Prob(x) >= 0.5:
		return 1
	else:
		return 0
"""
# Error Rate
count = 0
for i in range(num_win):
	if _Classfication(x_win[i,:]) != 1:
		count += 1
print(count)
for i in range(num_lose): 
	if _Classfication(x_lose[i,:]) != 0:
		count += 1

print("Count: %d | Error Rate: %f" %(count,(count/num_total)))
"""
############################################################################

test_data = pd.read_csv(sys.argv[3], encoding = 'UTF-8')
test_data = test_data.astype(float).as_matrix()
test_data = test_data[:,sel_array]
output = []
for i in range(test_data.shape[0]):
	outcome = _Classfication(test_data[i,:])
	output.append(outcome)

result=pd.DataFrame()
result['id'] = [str(i) for i in range(1,test_data.shape[0]+1)]
result['label'] = pd.Series(output, index = result.index)
result.to_csv(sys.argv[4],index=False)






