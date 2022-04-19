from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from numpy.random import seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt

from reader import read
from model import autoencoder_model

seed(10)
tf.random.set_seed(10)

#Read data.
data_path = "datasets/vulnbank_train.txt"
data_array = read(data_path)
print(data_array,"data_array")

#Divide data into train and test.
data_train,data_test = train_test_split(data_array,test_size=0.2,random_state=0)

print("Number of train datas:",data_train.shape[0])
print("Number of test datas:",data_test.shape[0])


# reshape inputs for LSTM [samples, timesteps, features]
X_train = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])
print("Test data shape:", X_test.shape)

# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
# model.summary()


# fit the model to the data
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.1).history

# plot the training losses
# fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
# ax.plot(history['loss'], 'b', label='Train', linewidth=2)
# ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
# ax.set_title('Model loss', fontsize=16)
# ax.set_ylabel('Loss (mae)')
# ax.set_xlabel('Epoch')
# ax.legend(loc='upper right')
# plt.show()


