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

# parameters
max_word = 20000
max_sequence_len = 50
embedding_vector_len = 150
threshold = 0.1
val = 0.1

# get labeled training data
data = pd.read_csv(sys.argv[1], sep = '\+\+\+\$\+\+\+', encoding = 'UTF-8', header = None, engine = 'python')
x_label = data[1].tolist()
y_label = data[0]
num_label = len(y_label)

#x_unlabel = pd.read_csv("training_nolabel.txt", sep = '\n', encoding = 'UTF-8', header = None)


x_test = pd.read_csv("testing_data.txt", sep = '\n', encoding = 'UTF-8',header = None, skiprows=1)
x_test_list = list()
for row in x_test[0]:
	x_test_list.append(row.split(',',1)[1])
x_test = x_test_list
del(x_test_list)

tokenizer = text.Tokenizer(num_words = max_word,filters='\t\n', split=' ')
tokenizer.fit_on_texts(x_label + x_test)
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# loading

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

x_label = tokenizer.texts_to_sequences(x_label)
#x_unlabel = t.texts_to_sequences(x_unlabel)
x_test = tokenizer.texts_to_sequences(x_test)
x_label = sequence.pad_sequences(x_label, maxlen = max_sequence_len)
#x_unlabel = sequence.pad_sequences(x_unlabel, maxlen = max_sequence_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_sequence_len)
x_label = np.asarray(x_label)
#x_unlabel = np.asarray(x_unlabel)
x_test = np.asarray(x_test)
"""
np.save('x_label.npy',x_label)
np.save('x_test.npy',x_test)
np.save('y_label.npy',y_label)

x_label = np.load('x_label.npy')
y_label = np.load('y_label.npy')
x_test = np.load('x_test.npy')
"""
num_label = len(y_label)
x_val = x_label[int(num_label*(1-val)):]
y_val = y_label[int(num_label*(1-val)):]
x_label = x_label[:int(num_label*(1-val))]
y_label = y_label[:int(num_label*(1-val))]
"""
model = Sequential()
model.add(Embedding(max_word+1, embedding_vector_len, input_length = max_sequence_len))#,batch_input_shape=(64,10)))
model.add(Dropout(0.5))

model.add(Conv1D(64,kernel_size=5,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=5,strides = 1))
model.add(Conv1D(128,kernel_size=5,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=5,strides = 1))
model.add(Conv1D(256,kernel_size=5,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=5,strides = 1))

#model.add(Flatten())
#model.add(MaxPooling1D(pool_size=4))
model.add(GRU(128,return_sequences=True))#,stateful=True,batch_input_shape=(64,10,50)))
model.add(Dropout(0.3))
model.add(Bidirectional(GRU(128,return_sequences=True)))#,stateful=True,batch_input_shape=(64,10,50)))
model.add(Dropout(0.3))
model.add(Bidirectional(GRU(128)))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
"""
model = Sequential()
model.add(Embedding(max_word+1, embedding_vector_len, input_length = max_sequence_len))
model.add(SpatialDropout1D(0.5))
model.add(Bidirectional(LSTM(256, recurrent_dropout=0.2, dropout=0.2), merge_mode='concat'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
model.summary()
checkpointer = ModelCheckpoint(filepath='0.8087RNN.h5',monitor = 'val_acc' ,save_best_only=True,save_weights_only=False, mode='max')
model.fit(x_label,y_label,batch_size=32,epochs=2,validation_data=(x_val,y_val), callbacks = [checkpointer])
for i in range(1):
    semi_pred = model.predict(x_test, batch_size=1024, verbose=True)
    semi_pred = np.squeeze(semi_pred)
    #index = (((semi_pred<1-threshold).astype(int)+(semi_pred>threshold).astype(int))-1).astype(bool)
    index = (semi_pred>1-threshold)+(semi_pred<threshold)
    semi_X = x_test
    semi_Y = (semi_pred > 0.5).astype(int)
    semi_X = semi_X[index,:]
    semi_Y = semi_Y[index] 
    semi_X = np.concatenate((semi_X, x_label))
    semi_Y = np.concatenate((semi_Y, y_label))
    model.fit(semi_X,semi_Y,batch_size=32,epochs=2,validation_data=(x_val,y_val), callbacks = [checkpointer])

"""
# testing
model = load_model('testRNN.h5')
predict = model.predict(x_test).reshape(-1,1)
np.save('predict.npy',predict)
predict = np.load('predict.npy')
predict = (predict > 0.5).astype(int).reshape(-1,1)

id_col = np.array([str(i) for i in range(predict.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, predict))
output = pd.DataFrame(data = output, columns = ['id', 'label'])
output.to_csv("predictRNN.csv", index = False)
"""
