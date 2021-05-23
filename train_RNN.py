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
from utils import load_csv, load_data, o_prep_scale
import os
import numpy as np


def init_stockpredict(timestep, feature_count):
    model = Sequential()
    model.add(LSTM(units=200, activation="relu", return_sequences=True, input_shape=(timestep,feature_count)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    sgd = Adam(lr=0.0001, decay=1e-6)
    model.summary()
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    return model


def train_model( symbol, dataset_loc, timestep, feature_count, epochs, feature_list):
    # load dataset
    dataset = load_csv(dataset_loc, symbol, timestep, feature_count, feature_list)
    dataset, scaler = o_prep_scale(np.nan_to_num(dataset[:, 1:dataset.shape[1]].astype(np.float64)), symbol)
    originaltrain, dates = load_data("train_data_RNN.csv", symbol, timestep, feature_count)
    train = scaler.transform(np.nan_to_num(originaltrain).astype(np.float64))
    train = np.delete(train,0,axis=0)
    trainX, trainY = np.reshape(train[:, :train.shape[1] - 1], (-1, timestep, feature_count)), train[:, train.shape[1] - 1]
    # initialize model
    model = init_stockpredict(timestep, feature_count)
    # fit model
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=64)
    modelpath = os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\models\\"
    print('Final Training Loss: %.7f' % history.history['loss'][len(history.history['loss']) - 1])
    save_model(model, modelpath + "20809984_RNN_model", overwrite=True)
