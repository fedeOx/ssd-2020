# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:04:26 2020

@author: Federico
"""
import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential # pip install keras
from keras.layers import Dense

def keras_reproducibility():
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
def train(train_x, train_y, hidden_layers, n_epochs, n_output=1):
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

def train_predict(model, train_x):
    trainPredict = model.predict(train_x)
    return trainPredict

def test_forecast(model, first_window, forecast_win_size):
    res = []
    actual_window = first_window
    for i in range(forecast_win_size):
        actual_window_resh = actual_window.reshape((1,len(first_window)))
        forecast = model.predict(actual_window_resh)
        res = np.append(res, [forecast])
        actual_window = actual_window[1:]
        actual_window = np.append(actual_window, forecast)
    return res 