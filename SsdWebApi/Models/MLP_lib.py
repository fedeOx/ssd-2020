# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:13:10 2020

@author: Federico
"""
import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def recursive_forecast(model, nparray, npred):
    nparray_resh = nparray.reshape((1,len(nparray)))
    forecast_elem = model.predict(nparray_resh)
    nparray = nparray[1:]
    new_nparray = np.append(nparray, [forecast_elem])
    if npred > 1:
        return np.append([forecast_elem], [recursive_forecast(model, new_nparray, npred-1)])
    else:
        return forecast_elem
    
"""
    Creates and train an MLP model.
    train_x -> the training set samples in sliding window format
    train_y -> the training set results
    test_x -> the test set samples in sliding window format
    test_y -> the test set results
    n_hidden -> defines hidden layers number and neurons number. ex: (100,) -> 1 hidden layer with 100 neurons
    n_epochs -> epochs number
"""
def train(train_x, train_y, test_x, test_y, n_hidden, n_epochs):
    # MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=n_hidden,
                    activation = 'relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    learning_rate='adaptive',
                    warm_start=True,
                    max_iter=1,
                    random_state=1234)
    print('Training ...')
    time_start = time.time()
    epochs_training_loss = []
    epochs_validation_r2 = []
    epochs_training_r2= []
    for i in range(n_epochs):
        mlp.fit(train_x, train_y)
        epochs_training_loss.append(mlp.loss_)
        epochs_training_r2.append(mlp.score(train_x, train_y) * 100)
        epochs_validation_r2.append(mlp.score(test_x, test_y) * 100)
        print("Epoch %2d: Loss = %5.4f, TrainR2 = %4.2f%%, ValidR2 = %4.2f%%" % (i+1, epochs_training_loss[-1], epochs_training_r2[-1], epochs_validation_r2[-1]))
    print('Total Time: %.2f sec' % (time.time() - time_start))
    max_valacc_idx = np.array(epochs_validation_r2).argmax()
    print('Max R2 on Validation = %4.2f%% at Epoch = %2d' % (epochs_validation_r2[max_valacc_idx], max_valacc_idx+1))
    return mlp

"""
    Uses an MLP model to make a prediction
    mlp -> the MLP model
    train_x -> the training set on which make predictions
    test_x -> the test set on which make predictions 
"""
def predict(mlp, train_x, test_x):
    trainPredict = mlp.predict(train_x) # predictions
    testForecast = mlp.predict(test_x) # forecast
    return trainPredict, testForecast

"""
    Uses an MLP model to make a forecast in the future
    mlp -> the MLP model
    test_x -> the test set samples in sliding window format
    forecast_win_size -> the number of elements to be predicted
"""
def forecast(mlp, test_x, forecast_win_size):
    last_elems = test_x[len(test_x)-1]
    return recursive_forecast(mlp, last_elems, forecast_win_size)