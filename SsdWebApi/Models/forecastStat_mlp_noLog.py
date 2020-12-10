# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:51:51 2020

@author: Federico
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, math, sys, io, base64

from keras.models import Sequential
from keras.layers import Dense

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
   dataset_forecast = dataset[-forecast_size:]
   
   cutpoint = int(0.7*len(dataset_base))
   train = dataset_base[:cutpoint]
   test = dataset_base[cutpoint:]
   print("Len train={0}, len test={1}".format(len(train), len(test)))
   
   # sliding window matrices (npast = window width)
   npast = 22*2 # 2 month considering working days only
   trainX, trainY = compute_windows(train, npast)
   testX, testY = compute_windows(test, npast)
   
   model = Sequential()
   n_hidden = 10
   n_output = 1
   model.add(Dense(n_hidden, input_dim=npast, activation='relu')) # hidden neurons, 1 layer
   model.add(Dense(n_output)) # output neurons
   model.compile(loss='mean_squared_error', optimizer='adam')
   model.fit(trainX, trainY, epochs=200, batch_size=128, verbose=2) # batch_size len(trainX)
   
   loss = model.history.history['loss']  
   plt.plot(range(len(loss)),loss);
   plt.ylabel('loss')
   plt.show()
   
   trainScore = model.evaluate(trainX, trainY, verbose=0)
   print('Score on train: MSE = {0:0.2f} '.format(trainScore))
   testScore = model.evaluate(testX, testY, verbose=0)
   print('Score on test: MSE = {0:0.2f} '.format(testScore))
   
   trainPredict = model.predict(trainX) # predictions
   testForecast = model.predict(testX) # forecast
   
   plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
   plt.plot(dataset_base)
   plt.plot(np.concatenate((np.full(1,np.nan),trainPredict[:,0])))
   plt.plot(np.concatenate((np.full(len(train),np.nan), testForecast[:,0])))
   plt.show()

   last_elems = testX[len(testX)-1]
   forecast_elems = my_predict(model, last_elems, forecast_size)
   
   plt.plot(dataset)
   plt.plot(trainPredict[:,0])
   plt.plot(np.concatenate((np.full(len(train),np.nan), testForecast[:,0])))
   plt.plot(np.concatenate((np.full(len(dataset_base),np.nan), forecast_elems)))
   plt.show()
  
   plt.plot(dataset_forecast)
   plt.plot(forecast_elems)
   plt.show()
   
   #plt.plot(df) # faccio il grafico dei dati contenuti nel file. Qui viene effettivamente generata un'immagine.
   #plt.show()
   
   # Finally, print the chart as base64 string to the console.
   #print_figure(plt.gcf()) # Qui viene chiamata la funzione print_figure passando in input plt.gcf(). gcf() va a prendere la figura corrente (ovvero quella creata nella riga sopra). Vedi commenti nella funzione.
   

   