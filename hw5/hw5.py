import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2
from keras.layers import Input, Embedding, Add, Dot, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import  sys

def load_data(fileName):
    df = pd.read_csv(fileName, encoding='utf8')

    user = df['UserID']
    user = np.array(user.astype(int))-1
    num_users = df['UserID'].drop_duplicates().max()

    movie = df['MovieID']
    movie = np.array(movie.astype(int))-1
    num_movies = df['MovieID'].drop_duplicates().max()

    rating = df['Rating']
    rating = rating.astype(float)

    return user, num_users, movie, num_movies, rating
def load_test_data(fileName):
    df = pd.read_csv(fileName, encoding='utf8')

    user = df['UserID']
    user = np.array(user.astype(int))-1
    num_users = df['UserID'].drop_duplicates().max()

    movie = df['MovieID']
    movie = np.array(movie.astype(int))-1
    num_movies = df['MovieID'].drop_duplicates().max()

    return user, num_users, movie, num_movies

#user, num_users, movie, num_movies, rating = load_data(sys.argv[1])
test_user, test_num_users, test_movie, test_num_movies = load_test_data(sys.argv[1])
#rating = rating / np.max(rating)

"""
dim = 128

u = Input(shape=(1,))
u_embedded = Embedding(num_users, dim)(u)
u_out = Dropout(0.5)(Flatten()(u_embedded))
m = Input(shape=(1,))
m_embedded = Embedding(num_movies, dim)(m)
m_out = Dropout(0.5)(Flatten()(m_embedded))

dot = Dot(axes=1)([u_out, m_out])

u_bias = Embedding(num_users, 1)(u)
u_bias = Flatten()(u_bias)
m_bias = Embedding(num_movies, 1)(m)
m_bias = Flatten()(m_bias)

out = Add()([dot, u_bias, m_bias])

model = Model(inputs=[u, m], outputs=out)
model.summary()

model.compile(loss='mse', optimizer='adam')
checkpoint = ModelCheckpoint(filepath='model.h5',monitor = 'val_loss' ,save_best_only=True,save_weights_only=False, mode='min')

idx = np.random.permutation(user.shape[0])
user = user[idx]
movie = movie[idx]
rating = rating[idx]

model.fit([user, movie], rating, 
                batch_size=512,
                epochs=50,
                validation_split=0.1,
                callbacks=[checkpoint])
"""
model = load_model('0.857.h5')
predict = model.predict([test_user,test_movie])
predict = np.clip(predict,1.0,5.0)

id_col = np.array([str(i+1) for i in range(predict.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, predict))
output = pd.DataFrame(data = output, columns = ['TestDataID', 'Rating'])
output.to_csv(sys.argv[2], index = False)



