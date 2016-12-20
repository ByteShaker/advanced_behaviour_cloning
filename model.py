import os
import json
import pickle
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

pickle_file = 'train_data.pickle'

bytes_in = bytearray(0)
max_bytes = 2 ** 31 - 1
input_size = os.path.getsize(pickle_file)

with open(pickle_file, 'rb') as p_train_data:
    for _ in range(0, input_size, max_bytes):
        bytes_in += p_train_data.read(max_bytes)
pickle_data = pickle.loads(bytes_in)
X_train = pickle_data['train_dataset']
y_train = pickle_data['train_labels']
X_val = pickle_data['val_dataset']
y_val = pickle_data['val_labels']
del pickle_data  # Free up memory

batch_size = 100
nb_classes = 1
nb_epoch = 40

X_train = X_train.astype('float32')
X_test = X_val.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')


#---Model-Definition:

input_shape = X_train.shape[1:]

model = Sequential()

model.add(Convolution2D(10, 5, 5, subsample=(5, 5), border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(Convolution2D(20, 3, 3, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(Convolution2D(30, 2, 2, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(Convolution2D(40, 2, 2, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='tf'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(40, name="hidden1"))
model.add(Activation('relu'))
model.add(Dense(20, name="hidden2"))
model.add(Activation('relu'))
model.add(Dense(10, name="hidden3"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, name="Steering_Angle"))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

json_string = model.to_json()
with open('./model.json', 'w') as outfile:
    json.dump(json_string, outfile)

model.save_weights('./model.h5')