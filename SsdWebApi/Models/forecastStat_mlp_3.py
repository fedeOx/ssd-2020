# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:31:40 2020

@author: Federico
"""
import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
   
   """
   plt.plot(logdata_base)
   plt.plot(np.concatenate((np.full(len(logdata_base),np.nan), logdata_forecast[:,0])))
   plt.show()
   """
   
   # Computes and stores the average vector and std dev for each feature
   scaler = StandardScaler().fit(dataset)   
   # Uses the average vector and std dev previously computed
   # to trasfor the dataset
   dataset_scaled = scaler.transform(dataset)
   
   forecast_size = 240*2 # 2 years considering working days only
   train = dataset_scaled[:-forecast_size]
   #train = np.log(train)
   test = dataset_scaled[-forecast_size:]
   #test = np.log(test)
   print("Len train={0}, len test={1}".format(len(train), len(test)))
   
   # sliding window matrices (npast = window width)
   npast = 22*2 # 2 month considering working days only
   train_x, train_y = compute_windows(train, npast)
   test_x, test_y = compute_windows(test, npast)
   
   mlp = MLPRegressor(hidden_layer_sizes=(100,100,),
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
   epochs_validation_accuracy = []
   epochs_training_accuracy = []
   max_epochs = 23
   for i in range(max_epochs):
       mlp.fit(train_x, train_y)
       epochs_training_loss.append(mlp.loss_)
       epochs_training_accuracy.append(mlp.score(train_x, train_y) * 100)
       epochs_validation_accuracy.append(mlp.score(test_x, test_y) * 100)
       print("Epoch %2d: Loss = %5.4f, TrainAccuracy = %4.2f%%, ValidAccuracy = %4.2f%%" % (i+1, epochs_training_loss[-1], epochs_training_accuracy[-1], epochs_validation_accuracy[-1]))
   print('Total Time: %.2f sec' % (time.time() - time_start))
   max_valacc_idx = np.array(epochs_validation_accuracy).argmax()
   print('Max Accuracy on Validation = %4.2f%% at Epoch = %2d' % (epochs_validation_accuracy[max_valacc_idx], max_valacc_idx+1))

   print("# livelli:", mlp.n_layers_)
   print("# neuroni di output:", mlp.n_outputs_)
   print("# neuroni nei livelli hidden:", mlp.get_params()["hidden_layer_sizes"])

   trainPredict = mlp.predict(train_x) # predictions
   testForecast = mlp.predict(test_x) # forecast
   
   plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
   plt.plot(dataset_scaled)
   plt.plot(np.concatenate((np.full(npast,np.nan),trainPredict)))
   plt.plot(np.concatenate((np.full(npast+len(train),np.nan), testForecast)))
   plt.show()

   last_elems = test_x[len(test_x)-1]
   forecast_elems_scaled = my_predict(mlp, last_elems, forecast_size)
   forecast_elems = scaler.inverse_transform(forecast_elems_scaled)
   
   plt.title(dffile[:-4])
   plt.plot(dataset)
   plt.plot(np.concatenate((np.full(npast,np.nan), scaler.inverse_transform(trainPredict))))
   plt.plot(np.concatenate((np.full(npast+len(train),np.nan), scaler.inverse_transform(testForecast))))
   plt.plot(np.concatenate((np.full(len(train)+len(test),np.nan), forecast_elems)))
   plt.show()
  
   """
   # recostruction
   plt.plot(dataset)
   plt.plot(np.exp(trainPredict))
   plt.plot(np.concatenate((np.full(len(train),np.nan), np.exp(testForecast[:,0]))))
   plt.plot(np.concatenate((np.full(len(dataset_base),np.nan), np.exp(forecast_elems))))
   plt.show()
   """
   plt.plot(scaler.inverse_transform(test))
   plt.plot(np.concatenate((np.full(len(test),np.nan),forecast_elems)))
   plt.show()
   
   #plt.plot(df) # faccio il grafico dei dati contenuti nel file. Qui viene effettivamente generata un'immagine.
   #plt.show()
   
   # Finally, print the chart as base64 string to the console.
   #print_figure(plt.gcf()) # Qui viene chiamata la funzione print_figure passando in input plt.gcf(). gcf() va a prendere la figura corrente (ovvero quella creata nella riga sopra). Vedi commenti nella funzione.