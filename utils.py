from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
from pickle import dump
import numpy as np
import pandas as pd
import json
# ********************* NOTE USING GENSIM LIBRARY TO REMOVE STOP WORDS*************************
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation


def load_csv(filename, symbol, timesteps, feature_count, feature_list):
    # The dataset is loaded from file
    dataset = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\data\\" + filename,
                              index_col=0, usecols=feature_list)
    dataset.index = pd.to_datetime(dataset.index)
    time = dataset.index.to_numpy(dtype='float64').reshape(len(dataset.index.values), 1)
    dataset = dataset.to_numpy()

    # duplicate dataset for each day and delete date first n rows from each  n where n is the number of days prior
    datasets = {}
    # time = dataset[:, [0]]
    datasets[0] = time
    output = dataset[:, [1]]
    datasets[1] = dataset[:-1]
    # remove date and closing price as these are no longer required
    datasets[1] = np.delete(datasets[1], [0], axis=1)
    for i in range(2, timesteps + 1):
        datasets[i] = datasets[i - 1][:-1]
    for i in range(1, timesteps + 1):
        # pad datasets with zeros to make sure the dimensions are aligned
        datasets[i] = np.append(np.zeros((i, feature_count)), datasets[i], axis=0)
    # concatenate all the datasets together
    dataset = np.append(datasets[0], datasets[1], axis=1)
    for i in range(1, len(datasets.keys()) - 1):
        dataset = np.concatenate((dataset, datasets[i + 1]), axis=1)
    # Move output to the last column
    dataset = np.concatenate((dataset, output), axis=1)
    # np.random.shuffle(dataset)
    # split test and train data 70-30
    train, test = dataset[:int(np.floor(0.7 * dataset.shape[0])), :], dataset[int(np.floor(0.3 * dataset.shape[0])):, :]
    # Save the train and test data files
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\data\\train_data_RNN.csv", train,
               delimiter=',',
               fmt='%s')
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\data\\test_data_RNN.csv", test,
               delimiter=',', fmt='%s')
    return dataset

    # load data from the preprocessed train and test data files


def load_data(filename, symbol, timestep, feature_count):
    dataset = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\data\\" + filename,
                         dtype=np.float,
                         delimiter=',', usecols=tuple(range(1, feature_count * timestep + 2)))
    dates = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\data\\" + filename,
                       delimiter=',', usecols=(0))
    # inputs,outputs = dataset[:,:dataset.shape[1]-1],dataset[:,dataset.shape[1]-1]
    return dataset, dates


# scale dataset between 0 and 1 using MinMax Scaler function and save scaler as a pickle file
def o_prep_scale(sets, symbol):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(sets)
    scaler_filename = "scaler.pkl"
    modelpath = os.path.dirname(os.path.realpath(__file__)) + "\\" + symbol + "\models\\"
    dump(sc, open(modelpath + scaler_filename, 'wb'))
    return training_set_scaled, sc
