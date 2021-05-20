# import required packages


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from utils import load_csv,load_data,o_prep_scale
import os
import numpy as np

def init_stockpredict():
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(3,4)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def run_test_harness(timestep, symbol, dataset_loc):
    # load dataset
    dataset = load_csv(dataset_loc, symbol, timestep)
    dataset, scaler = o_prep_scale(dataset[:, 1:dataset.shape[1]])
    originaltrain, dates = load_data("train_data_RNN.csv", symbol,timestep)
    train = scaler.transform(originaltrain)
    trainX, trainY = np.reshape(train[:, :train.shape[1] - 1], (-1, timestep, 88)), train[:, train.shape[1] - 1]
    # initialize model
    model = init_stockpredict()
    # fit model
    history = model.fit(trainX, trainY, epochs=50, batch_size=32)
    modelpath = os.path.dirname(os.path.realpath(__file__)) + "\\"+ symbol + "\models\\"
    print('Final Training Loss: %.7f'% history.history['loss'][len(history.history['loss'])-1])
    save_model(model, modelpath + "20809984_RNN_model", overwrite= True)