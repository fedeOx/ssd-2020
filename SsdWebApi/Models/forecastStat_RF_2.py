# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:10:37 2020

@author: Federico
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, math, sys, io, base64

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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
    forecast_elem = model.predict(nparray_resh)
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
   
   forecast_size = 120 # 2 years considering working days only
   dataset_base = dataset[:-forecast_size]
   #logdata_base = np.log(dataset_base)
   dataset_forecast = dataset[-forecast_size:]
   #logdata_forecast = np.log(dataset_forecast)
   
   # transform the time series data into supervised learning
   n_in = 22 # 1 month considering working days only
   data = series_to_supervised(dataset_base, n_in)
   
   train_perc = 0.7
   # split dataset
   train, test = train_test_split(data, train_perc)
   # transform list into array
   train = np.asarray(train)
   # split into input and output columns
   train_x, train_y = train[:, :-1], train[:, -1]
   test_x, test_y = test[:, :-1], test[:, -1]
   # fit model
   model = RandomForestRegressor()
   param_grid = [{'n_estimators': [10, 20, 50], 'max_depth': [5, 10, 15]}]
   gs_model = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
   gs_model.fit(train_x, train_y)
   
   # Stampa risultati
   print('Combinazioni di parametri:\n', gs_model.cv_results_['params'])
   print('Accuratezza media per combinazione:\n', np.sqrt(-gs_model.cv_results_['mean_test_score']))
   print('Combinazione migliore:\n', gs_model.best_params_)
   print('Accuratezza media della combinazione migliore: %.3f' % np.sqrt(-gs_model.best_score_))
   
   model = RandomForestRegressor(**gs_model.best_params_)
   print('Addestramento in corso ...')
   model.fit(train_x, train_y)
   
   # Ottenimento delle predizioni (train) e calcolo RMSE
   train_y_predicted = model.predict(train_x)
   rmse = np.sqrt(mean_squared_error(train_y, train_y_predicted))
   print('Train RMSE: ', rmse)
   # Ottenimento delle predizioni (validation) e calcolo RMSE
   test_y_predicted = model.predict(test_x)
   rmse = np.sqrt(mean_squared_error(test_y, test_y_predicted))
   print('Validation RMSE: ', rmse)
   
   print('R2 score:', model.score(test_x, test_y))
   
   plt.figure(figsize=(14, 10))
   plt.title(dffile[:-4])
   plt.plot(dataset_base)
   plt.plot(np.concatenate((np.full(n_in,np.nan), train_y_predicted)))
   plt.plot(np.concatenate((np.full(n_in+len(train_y_predicted)+n_in,np.nan), test_y_predicted)))
   plt.show()

   last_elems = test_x[len(test_x)-1]
   forecast_elems = my_predict(model, last_elems, forecast_size)
   
   plt.plot(dataset)
   plt.plot(train_y_predicted)
   plt.plot(np.concatenate((np.full(len(train_y_predicted),np.nan), test_y_predicted)))
   plt.plot(np.concatenate((np.full(len(dataset_base),np.nan), forecast_elems)))
   plt.show()

   plt.plot(dataset_forecast)
   plt.plot(forecast_elems)
   plt.show()
   
   #plt.plot(df) # faccio il grafico dei dati contenuti nel file. Qui viene effettivamente generata un'immagine.
   #plt.show()
   
   # Finally, print the chart as base64 string to the console.
   #print_figure(plt.gcf()) # Qui viene chiamata la funzione print_figure passando in input plt.gcf(). gcf() va a prendere la figura corrente (ovvero quella creata nella riga sopra). Vedi commenti nella funzione.