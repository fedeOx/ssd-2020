# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:04:26 2020

@author: Federico
"""
import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

class MLPUtil:
    def __init__(self, dataset_scaled, sliding_win_size, forecast_win_size, ext_vars_time):
        self.dataset_scaled = dataset_scaled
        self.sliding_win_size = sliding_win_size
        self.forecast_win_size = forecast_win_size
        self.ext_vars_time = ext_vars_time
        self.trainPredict_scaled = []
        self.forecast_scaled = []
        self._keras_reproducibility()

    def _keras_reproducibility(self):
       # Seed value
       seed_value=1234
       # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
       os.environ['PYTHONHASHSEED']=str(seed_value)
       # 2. Set the `python` built-in pseudo-random generator at a fixed value
       random.seed(seed_value)
       # 3. Set the `numpy` pseudo-random generator at a fixed value
       np.random.seed(seed_value)
       # 4. Set the `tensorflow` pseudo-random generator at a fixed value
       tf.compat.v1.set_random_seed(seed_value)
       # 5. Configure a new global `tensorflow` session
       session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
       sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
       tf.compat.v1.keras.backend.set_session(sess)
   
    """
        Creates and train an MLP model.
        train_x -> the training set samples in sliding window format
        train_y -> the training set results
        hidden_layers -> defines hidden layers number and neurons number. ex: (100,) -> 1 hidden layer with 100 neurons
        n_epochs -> epochs number
    """
    def train(self, train_x, train_y, hidden_layers, n_epochs, n_output=1):
        model = Sequential()
        for n_neurons in hidden_layers:
            model.add(Dense(n_neurons, activation="relu"))
        model.add(Dense(n_output))
        opt = keras.optimizers.Adam(learning_rate=0.002)
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=len(train_x), verbose=2) # batch_size len(trainX)
        # Model performance
        trainScore = model.evaluate(train_x, train_y, verbose=0)
        print('Score on train: MSE = {0} '.format(trainScore))
        return model

    def train_predict(self, model, train_x):
        trainPredict = model.predict(train_x)
        self.trainPredict_scaled = trainPredict
        return trainPredict

    def test_forecast(self, model, first_window, ext_vars_tuple):
        res = []
        actual_window = first_window
        j = 0 # index from which start reading ext_data values
        k = self.ext_vars_time # how many times repeat j-th ext_data value
        for i in range(self.forecast_win_size):
            for i2 in range(len(ext_vars_tuple)):
                actual_window = np.append(actual_window, [ext_vars_tuple[i2][j]]) # add ext vars values
            k -= 1
            if k == 0:
                j += 1
                k = self.ext_vars_time 
            actual_window_resh = actual_window.reshape((1,len(first_window)+len(ext_vars_tuple)))
            forecast = model.predict(actual_window_resh)
            res = np.append(res, [forecast])
            actual_window = actual_window[1:] # remove head
            actual_window = actual_window[:-len(ext_vars_tuple)] # remove ext vars values
            actual_window = np.append(actual_window, forecast)
        self.forecast_scaled = res
        return res 

    def plotResult(self, plot_name):
        plt.rcParams["figure.figsize"] = (16,10)
        plt.title(plot_name)
        plt.plot(self.dataset_scaled)
        plt.plot(np.concatenate((np.full(self.sliding_win_size, np.nan), self.trainPredict_scaled[:,0])))
        plt.plot(np.concatenate((np.full(len(self.dataset_scaled)-self.forecast_win_size, np.nan), self.forecast_scaled)))
        plt.show()

    def plotZoom(self, plot_name, test):
        plt.rcParams["figure.figsize"] = (16,10)
        plt.title(plot_name)
        plt.plot(test)
        plt.plot(self.forecast_scaled)
        plt.show()
