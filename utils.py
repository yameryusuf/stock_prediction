from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
from pickle import dump
import numpy as np
import json
from conversion_file import conversion
# ********************* NOTE USING GENSIM LIBRARY TO REMOVE STOP WORDS*************************
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
def load_csv(filename, symbol, timesteps):
    # The dataset is loaded from file
    dataset = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\"+ symbol  + "\data\\" + filename, skiprows=1,
                         delimiter=',', usecols=tuple(range(0,88)), dtype=object,
                         converters=conversion)

    # duplicate dataset for each day and delete date first n rows from each  n where n is the number of days prior
    datasets = {}
    time = dataset[:, [0]]
    datasets[0] = time
    output = dataset[:, [1]]
    datasets[1] = np.delete(dataset, 0, axis=0)
    # pad datasets with zeros to make sure the dimensions are aligned
    datasets[1] = np.append(datasets[1], np.zeros((1, 6)), axis=0)
    # remove date and closing price as these are no longer required
    datasets[1] = np.delete(datasets[1], [0, 1], axis=1)
    for i in range(2,timesteps):
        datasets[i] = np.delete(dataset[i-1], 0, axis=0)
        # pad datasets with zeros to make sure the dimensions are aligned
        datasets[i] = np.append(datasets[i], np.zeros((i, 6)), axis=0)
        # remove date and closing price as these are no longer required
        datasets[i] = np.delete(datasets[i], [0, 1], axis=1)
    # concatenate all the datasets together


    for i in range(len(datasets.keys())-1):
        dataset = np.concatenate((datasets[i], dataset[i+1]), axis=1)
    # Move output to the last column
    np.concatenate((dataset, output), axis=1)
    np.random.shuffle(dataset)
    # split test and train data 70-30
    train, test = dataset[:int(np.floor(0.7 * dataset.shape[0])), :], dataset[int(np.floor(0.7 * dataset.shape[0])):, :]
    # Save the train and test data files
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\data\\train_data_RNN.csv", train, delimiter=',',fmt='%s')
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\data\\test_data_RNN.csv", test, delimiter=',', fmt='%s')
    return dataset

    # load data from the preprocessed train and test data files


def load_data(filename, symbol, timestep):
    dataset = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\"+ symbol + "\data\\" + filename, dtype=np.float,
                         delimiter=',', usecols=tuple(range(0,88*(timestep)+2)))
    dates = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\"+ symbol + "\data\\" + filename,
                       delimiter=',', usecols=(0), dtype=object,
                       converters={0: datetime.strptime(x.decode('ascii'), "%Y-%m-%d %H:%M:%S"),
                                   })
    # inputs,outputs = dataset[:,:dataset.shape[1]-1],dataset[:,dataset.shape[1]-1]
    return dataset, dates

# scale dataset between 0 and 1 using MinMax Scaler function and save scaler as a pickle file
def o_prep_scale(sets):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(sets)
    scaler_filename = "scaler.pkl"
    modelpath = os.path.dirname(os.path.realpath(__file__)) + "\\"+ symbol + "\models\\"
    dump(sc, open(modelpath + scaler_filename, 'wb'))
    return training_set_scaled, sc