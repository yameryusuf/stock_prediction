# import required packages
from tensorflow.keras.models import load_model
import datetime as datetime
from utils import load_csv,load_data,o_prep_scale
import os
import numpy as np
import matplotlib.pyplot as plt
from pickle import load
import time as time
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
    modelpath = os.path.dirname(os.path.realpath(__file__)) + "\models\\20809984_RNN_model"
    model = load_model(modelpath)
    scalerpath = os.path.dirname(os.path.realpath(__file__)) + "\models\\scaler.pkl"
    scaler = load(open(scalerpath, 'rb'))
    # 2. Load your testing data
    originaltest, dates = load_data("test_data_RNN.csv")
    real_stock_price = originaltest[:, originaltest.shape[1] - 1]
    test = scaler.transform(originaltest)
    testX, testY = np.reshape(test[:, :test.shape[1] - 1], (-1, 3, 4)), test[:, test.shape[1] - 1]
	# 3. Run prediction on the test data and output required plot and loss
    predicted_stock_price = scaler.inverse_transform(np.concatenate((test[:, :test.shape[1] - 1],model.predict(testX)), axis=1))[:, test.shape[1] - 1]
    # combine test input dataset with predicted and real stock price for plotting
    loadfile = np.concatenate((dates.reshape(-1,1), originaltest,predicted_stock_price.reshape(-1,1)), axis=1)
    # sort by date
    loadfile = loadfile[np.argsort(loadfile[:, 0])]
    loss, acc = model.evaluate(testX, testY, verbose=0)
    print('>Final Testing Loss: %3f'% loss)
    print('>Final Testing Accuracy: %3f'% ((1-np.average(abs(predicted_stock_price-real_stock_price)/predicted_stock_price))*100))
    # Plot the stock price against date
    fig, ax = plt.subplots(figsize=(10, 10))
    dates = [x.strftime("%m/%d/%Y") for x in list(loadfile[:, 0])]
    ax.plot(dates,loadfile[:,-2], color='black', label='Stock Price')
    ax.plot(dates,loadfile[:,-1], color='green', label='Predicted Stock Price')
    ax.set_xticks(dates[::30])
    ax.set_xticklabels(dates[::30], rotation=45)
    ax.set(title='Test Stock Prediction', xlabel='Time',ylabel ='Stock Price')
    ax.legend()
    plt.show()
