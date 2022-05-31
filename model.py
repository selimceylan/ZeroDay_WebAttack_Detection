from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Bidirectional
from keras.models import Model
from keras import regularizers
import keras
import numpy as np


# define the autoencoder network model
# def autoencoder_model(X):
#     inputs = Input(shape=(X.shape[1], X.shape[2]))
#     L1 = LSTM(16, activation='relu', return_sequences=True,
#               kernel_regularizer=regularizers.l2(0.00))(inputs)
#     L2 = LSTM(4, activation='relu', return_sequences=False,kernel_regularizer=Dropout(0.2))(L1)
#     L3 = RepeatVector(X.shape[1])(L2)
#     L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
#     L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
#     output = TimeDistributed(Dense(X.shape[2]))(L5)
#     model = Model(inputs=inputs, outputs=output)
#     return model



def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(128, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(64, activation='relu', return_sequences=True)(L1)
    L3 = LSTM(32, activation='relu', return_sequences=True)(L2)
    L4 = LSTM(16, activation='relu', return_sequences=False)(L3)
    L5 = RepeatVector(X.shape[1])(L4)
    L6 = LSTM(16, activation='relu', return_sequences=True)(L5)
    L7 = LSTM(32, activation='relu', return_sequences=True)(L6)
    L8 = LSTM(64, activation='relu', return_sequences=True)(L7)
    L9 = LSTM(128, activation='relu', return_sequences=True)(L8)
    output = TimeDistributed(Dense(X.shape[2]))(L9)
    model = Model(inputs=inputs, outputs=output)
    return model
