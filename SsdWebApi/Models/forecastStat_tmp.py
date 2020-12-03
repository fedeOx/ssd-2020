# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:52:41 2020

"""

import sys, io, base64
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import pmdarima as pm # pip install pmdarima
from statsmodels.tsa.stattools import acf

def print_figure(fig):
	"""
	Converts a figure (as created e.g. with matplotlib or seaborn) to a png image and this 
	png subsequently to a base64-string, then prints the resulting string to the console.
	"""
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	print(base64.b64encode(buf.getbuffer()))


if __name__ == "__main__":
   # change working directory to script path
   abspath = os.path.abspath(__file__)
   dname = os.path.dirname(abspath)
   os.chdir(dname)

   print('MAPE Number of arguments:', len(sys.argv)) # Scrive la lunghezza del vettore degli argomenti (argv).
   print('MAPE Argument List:', str(sys.argv), ' first true arg:',sys.argv[1]) # Scrive la lista degli argomenti seguito dal secondo argomento di argv che sarebbe il file csv (il primo è il file pythos stesso).
  
   dffile = sys.argv[1] # recupero il file che voglio andare a leggere
   df = pd.read_csv("../" + dffile) # leggo il contenuto del file
   
   values = df[dffile[:-4]].to_numpy() # array of sales data
   logdata = np.log(values) # log transform
   data = pd.Series(logdata).diff() # convert to pandas series
   
   plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
   plt.plot(data.values); plt.show() # data plot
   

   # acf plot, industrial
   import statsmodels.api as sm
   sm.graphics.tsa.plot_acf(data.values, lags=261*8)
   plt.show()
   
   # train and test set
   #train = data[:-12]
   #test = data[-12:]
   
   # simple reconstruction, not necessary, unused
   #reconstruct = np.exp(np.r_[train,test])

   
   plt.plot(df); plt.show()
   
   # Finally, print the chart as base64 string to the console.
   print_figure(plt.gcf())
   

   