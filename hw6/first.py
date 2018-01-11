import sys
import csv
import numpy as np
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Add, Dot, Flatten, Conv2D, MaxPooling2D, Reshape,Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans

encoding_dim = 64
name = 'first' 

ori_imag = np.load(sys.argv[1]).astype(float)
ori_imag = ori_imag/255.0

"""
input_img = Input(shape=(784,), name = 'imp')
input_reshape = Reshape((28,28,1))(input_img)
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_reshape)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
encoded_flatten = (Flatten()(x))
encoded_dense = (Dense(128, activation='relu')(encoded_flatten))
encoded_dense = (Dense(128, activation='relu')(encoded_dense))
encoded = Dense(encoding_dim, activation='relu', name = 'out')(encoded_dense)

decoded_dense = (Dense(256, activation='relu')(encoded))
decoded_dense = (Dense(256, activation='relu')(decoded_dense))
decoded_dense = (Dense(256, activation='relu')(decoded_dense))
decoded = Dense(784, activation='sigmoid')(decoded_dense)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
checkpoint = ModelCheckpoint(name+'best.h5', monitor = 'val_loss', save_best_only = True)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.summary()
autoencoder.fit(ori_imag, ori_imag, epochs=30, batch_size=512, shuffle=True, validation_split = 0.1, callbacks = [checkpoint])

encoder.save('first.h5')
"""
encoder = load_model('first.h5')

# Kmeans
low_dim = encoder.predict(ori_imag)

test1 = KMeans(n_clusters = 64, max_iter = 300).fit(low_dim)

center1 = test1.cluster_centers_
test2 = KMeans(n_clusters = 16, max_iter = 10).fit(center1)

center2 = test2.cluster_centers_
test3 = KMeans(n_clusters = 2, max_iter = 10).fit(center2)

testing = pd.read_csv(sys.argv[2])
imageOne = low_dim[np.array(testing['image1_index']).reshape(-1, 1)].reshape(-1, encoding_dim)
imageTwo = low_dim[np.array(testing['image2_index']).reshape(-1, 1)].reshape(-1, encoding_dim)

result1 = test3.predict(center2[test2.predict(center1[test1.predict(imageOne)])])
result2 = test3.predict(center2[test2.predict(center1[test1.predict(imageTwo)])])


output_temp = (result1==result2).astype(int).reshape(-1, )

output = pd.DataFrame()
output['ID'] = [str(i) for i in range(output_temp.shape[0])]
output['Ans'] = pd.Series(output_temp, index = output.index)
output.to_csv(sys.argv[3], index=False)


