import numpy as np
import pandas as pd 
import sys
from keras.preprocessing import sequence, text
from keras.layers.embeddings import Embedding
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Activation, SpatialDropout1D
from keras.layers import Bidirectional,GRU, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, BatchNormalization, LSTM
from keras.optimizers import SGD, Adam
from keras.initializers import he_normal, he_uniform
from keras.callbacks import ModelCheckpoint
import pickle

max_word = 20000
max_sequence_len = 50
embedding_vector_len = 150
threshold = 0.1
val = 0.1
"""
data = pd.read_csv("training_label.txt", sep = '\+\+\+\$\+\+\+', encoding = 'UTF-8', header = None, engine = 'python')
x_label = data[1].tolist()
y_label = data[0]
num_label = len(y_label)
"""
x_test = pd.read_csv(sys.argv[1], sep = '\n', encoding = 'UTF-8',header = None, skiprows=1)
x_test_list = list()
x_testBOW_list = list()
for row in x_test[0]:
	x_test_list.append(row.split(',',1)[1])
	x_testBOW_list.append(row.split(',',1)[1])
x_test = x_test_list
x_testBOW = x_testBOW_list
del(x_test_list)
del(x_testBOW_list)
"""
tokenizer = text.Tokenizer(num_words = max_word)
tokenizer.fit_on_texts(x_label + x_test)
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# loading
"""
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
with open('tokenizerBOW.pickle', 'rb') as handle:
	tokenizerBOW = pickle.load(handle)

x_testBOW = tokenizerBOW.texts_to_matrix(x_testBOW)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen = max_sequence_len)
x_test = np.asarray(x_test)

#model1 = load_model('0.805RNN.h5')
model2 = load_model('0.8087RNN.h5')
model3 = load_model('0.808CNN.h5')
#model4 = load_model('0.809RNN.h5')
#model5 = load_model('0.80BOW.h5')
print("finish loading model")

#predict1 = model1.predict(x_test).reshape(-1,1)
predict2 = model2.predict(x_test).reshape(-1,1)
predict3 = model3.predict(x_test).reshape(-1,1)
#predict4 = model3.predict(x_test).reshape(-1,1)
#predict5 = model4.predict(x_testBOW).reshape(-1,1)

"""
predict1 = np.load('predict0.805RNN.npy')
predict2 = np.load('predict0.8087RNN.npy')
predict3 = np.load('predict0.808CNN.npy')
predict4 = np.load('predict0.809RNN.npy')
predict5 = np.load('predict0.80BOW.npy')
"""
#predict = 0.1 * predict1 + 0.2 * predict2 + 0.15 * predict3 + 0.3 * predict4 + 0.25 * predict5
predict = 0.5 * predict2 + 0.5 * predict3
predict = (predict > 0.5).astype(int).reshape(-1,1)

# generate prediction file
id_col = np.array([str(i) for i in range(predict.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, predict))
output = pd.DataFrame(data = output, columns = ['id', 'label'])
output.to_csv(sys.argv[2] , index = False)
