import numpy as np
import pandas as pd
import sys
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential,load_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.initializers import he_normal, he_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

raw_data = pd.read_csv(sys.argv[1], sep = ',' ,encoding = 'UTF-8')

raw_data = raw_data.as_matrix()
y_data = raw_data[:,0]
x_data = list()
num = len(y_data)
for i in range(num):
	temp = raw_data[i][1].split()
	x_data.append(temp)
x_data = np.array(x_data).astype(float).reshape(num,48,48,1)/255.0
y_data = np_utils.to_categorical(y_data,7)
"""
np.save('x_data.npy',x_data)
np.save('y_data.npy',y_data)

x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')
"""
num = len(y_data)
x_train = x_data[:int(0.9*num),:,:,:]
y_train = y_data[:int(0.9*num),:]
num_train = x_train.shape[0]
x_val = x_data[int(0.9*num):,:,:,:]
y_val = y_data[int(0.9*num):,:]
num_val = x_val.shape[0]

model = Sequential() # kernel_constraint

model.add(Conv2D(32,kernel_size = (3,3),input_shape=(48,48,1),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32,kernel_size = (3,3),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size = (3,3),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64,kernel_size = (3,3),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size = (3,3),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128,kernel_size = (3,3),padding='same',kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(7,activation='softmax'))

model.summary();
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.001), metrics=['accuracy'])

datagen = ImageDataGenerator(
    			featurewise_center=False,
    			featurewise_std_normalization=False,
    			rotation_range=15,
    			width_shift_range=0.2,
    			height_shift_range=0.2,
    			shear_range=0.2,
			zoom_range=0.2,
    			horizontal_flip=True)
datagen.fit(x_train)
for i in range(40):
	checkpointer = ModelCheckpoint(filepath='0.68.h5',monitor = 'val_acc' ,save_best_only=True,save_weights_only=False, mode='max')
	model.fit(x_train,y_train,batch_size=128,epochs=1,validation_data=(x_val,y_val))
	model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),num_train/32,epochs=5,validation_data=(x_val,y_val),validation_steps=num_val/32, callbacks=[checkpointer])

################################################################################################
"""
model = load_model('0.68.h5')

test_data = pd.read_csv(sys.argv[2], sep = ',', encoding = 'UTF-8')
test_data = test_data.as_matrix()
id_data = test_data[:,0]
test_num = len(id_data)
test = list()
for i in range(test_num):
	temp = test_data[i][1].split()
	test.append(temp)
test = np.array(test).astype(float).reshape(test_num,48,48,1)/255.0

result = model.predict(test)
result = result.argmax(axis=-1)

output=pd.DataFrame()
output['id'] = [str(i) for i in range(test_num)]
output['label'] = pd.Series(result, index = output.index)
output.to_csv('firstCNN.csv',index=False)
"""
