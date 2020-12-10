# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:10:37 2020

@author: Federico
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, math, sys, io, base64

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def print_figure(fig):
	"""
	Converts a figure (as created e.g. with matplotlib or seaborn) to a png image and this 
	png subsequently to a base64-string, then prints the resulting string to the console.
	"""
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	print(base64.b64encode(buf.getbuffer()))

def compute_windows(nparray, npast=1):
    dataX, dataY = [], [] # window and value
    for i in range(len(nparray)-npast-1):
        a = nparray[i:(i+npast), 0]
        dataX.append(a)
        dataY.append(nparray[i + npast, 0])
    return np.array(dataX), np.array(dataY)

def my_predict(model, nparray, npred):
    nparray_resh = nparray.reshape((1,len(nparray)))
    forecast_elem = model.predict(nparray_resh, verbose=0)
    nparray = nparray[1:]
    new_nparray = np.append(nparray, [forecast_elem])
    if npred > 1:
        return np.append([forecast_elem], [my_predict(model, new_nparray, npred-1)])
    else:
        return forecast_elem

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, train_perc):
    cutpoint = int(train_perc*len(data))
    train = data[:cutpoint]
    test = data[cutpoint:]
    return train, test

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

if __name__ == "__main__":
   # change working directory to script path
   abspath = os.path.abspath(__file__)
   dname = os.path.dirname(abspath)
   os.chdir(dname)

   print('MAPE Number of arguments:', len(sys.argv)) # Scrive la lunghezza del vettore degli argomenti (argv).
   print('MAPE Argument List:', str(sys.argv), ' first true arg:',sys.argv[1]) # Scrive la lista degli argomenti seguito dal secondo argomento di argv che sarebbe il file csv (il primo è il file pythos stesso).
   
   dffile = sys.argv[1] # recupero il file che voglio andare a leggere
   df = pd.read_csv("../"+dffile) # leggo il contenuto del file
   dataset = df.values
   dataset = dataset.astype('float32')
   
   plt.plot(dataset)
   plt.show()
   
   forecast_size = 240*2 # 2 years considering working days only
   dataset_base = dataset[:-forecast_size]
   #logdata_base = np.log(dataset_base)
   dataset_forecast = dataset[-forecast_size:]
   #logdata_forecast = np.log(dataset_forecast)
   
   # transform the time series data into supervised learning
   n_in = 22 # 1 month considering working days only
   data = series_to_supervised(dataset_base, n_in)
   
   # walk-forward validation for univariate data
   predictions = list()
   train_perc = 0.7
   # split dataset
   train, test = train_test_split(data, train_perc)
   # seed history with training dataset
   history = [x for x in train]
   # step over each time-step in the test set
   for i in range(len(test)):
       # split test row into input and output columns
       testX, testy = test[i, :-1], test[i, -1]
       # fit model on history and make a prediction
       yhat = random_forest_forecast(history, testX)
       # store forecast in list of predictions
       predictions.append(yhat)
       # add actual observation to history for the next loop
       history.append(test[i])
       # summarize progress
       print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
   # estimate prediction error
   mae = mean_absolute_error(test[:, -1], predictions)
   y = test[:, -1]
   ythat = predictions
   print('MAE: %.3f' % mae)
   # plot expected vs predicted
   plt.plot(y, label='Expected')
   plt.plot(yhat, label='Predicted')
   plt.legend()
   plt.show()
   
   #plt.plot(logdata_base)
   #plt.plot(np.concatenate((np.full(len(logdata_base),np.nan), logdata_forecast[:,0])))
   #plt.show()
   
   """
   cutpoint = int(0.7*len(logdata_base))
   train = logdata_base[:cutpoint]
   test = logdata_base[cutpoint:]
   print("Len train={0}, len test={1}".format(len(train), len(test)))
   
   # sliding window matrices (npast = window width)
   npast = 22*2 # 2 month considering working days only
   trainX, trainY = compute_windows(train, npast)
   testX, testY = compute_windows(test, npast)
   
   from keras.wrappers.scikit_learn import KerasRegressor
   model = KerasRegressor(build_fn=create_model, verbose=0)
   
   from sklearn.model_selection import GridSearchCV
   # define the grid search parameters
   batch_size = [10, 20, 40, 60, 80, 100, 200, 300]
   epochs = [10, 50, 100]
   param_grid = dict(batch_size=batch_size, epochs=epochs)
   grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
   grid_result = grid.fit(trainX, trainY)
   # summarize results
   bestBatchSize = grid_result.best_params_['batch_size']
   bestEpochs = grid_result.best_params_['epochs']
   print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
   means = grid_result.cv_results_['mean_test_score']
   stds = grid_result.cv_results_['std_test_score']
   params = grid_result.cv_results_['params']
   for mean, stdev, param in zip(means, stds, params):
       print("%f (%f) with: %r" % (mean, stdev, param))
   
   print('Fitting using batch_size={0} and epochs={1}'.format(bestBatchSize, bestEpochs))
   model.fit(trainX, trainY, epochs=bestEpochs, batch_size=bestBatchSize, verbose=2) # batch_size len(trainX)
   
   trainPredict = model.predict(trainX) # predictions
   testForecast = model.predict(testX) # forecast
   
   plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
   plt.plot(np.log(dataset_base))
   plt.plot(np.concatenate((np.full(1,np.nan),trainPredict)))
   plt.plot(np.concatenate((np.full(len(train)+1,np.nan), testForecast)))
   plt.show()

   last_elems = testX[len(testX)-1]
   forecast_elems = my_predict(model, last_elems, forecast_size)
   
   plt.plot(np.log(dataset))
   plt.plot(trainPredict)
   plt.plot(np.concatenate((np.full(len(train),np.nan), testForecast)))
   plt.plot(np.concatenate((np.full(len(dataset_base),np.nan), forecast_elems)))
   plt.show()
  
   # recostruction
   plt.plot(dataset)
   plt.plot(np.exp(trainPredict))
   plt.plot(np.concatenate((np.full(len(train),np.nan), np.exp(testForecast))))
   plt.plot(np.concatenate((np.full(len(dataset_base),np.nan), np.exp(forecast_elems))))
   plt.show()

   plt.plot(np.exp(logdata_forecast))
   plt.plot(np.exp(forecast_elems))
   plt.show()
   
   #plt.plot(df) # faccio il grafico dei dati contenuti nel file. Qui viene effettivamente generata un'immagine.
   #plt.show()
   
   # Finally, print the chart as base64 string to the console.
   #print_figure(plt.gcf()) # Qui viene chiamata la funzione print_figure passando in input plt.gcf(). gcf() va a prendere la figura corrente (ovvero quella creata nella riga sopra). Vedi commenti nella funzione.
   """