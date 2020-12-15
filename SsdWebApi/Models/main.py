# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:06:42 2020

@author: Federico
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys

from sklearn.preprocessing import StandardScaler
import keras_MLP_lib as myMLP
import my_utils as utils
import PSO as myPSO
from sklearn.metrics import mean_squared_error

forecast_win_size = 264 # 1 years considering working days only
sliding_win_size = 22*6 # 2 month considering working days only

def make_MLP_forecast(index_name, dataset_scaled, hidden_layers, n_epochs):
    myMLP.keras_reproducibility()
    
    train, test = utils.test_train_split(dataset_scaled, forecast_win_size)

    # sliding window matrices
    train_x, train_y = utils.series_to_supervised(train, sliding_win_size)
    
    model_t = myMLP.train(train_x, train_y, hidden_layers, n_epochs)
    
    trainPredict_scaled = myMLP.train_predict(model_t, train_x)
    
    last = train_x[len(train_x)-1]
    last = last[1:]
    first_window = np.append(last, train_y[-1])
    forecast_scaled = myMLP.test_forecast(model_t, first_window, forecast_win_size)
    print('Score on test: MSE = {0}'.format(mean_squared_error(test, forecast_scaled)))
    
    plt.rcParams["figure.figsize"] = (16,10)
    plt.title(index_name)
    plt.plot(dataset_scaled)
    plt.plot(np.concatenate((np.full(sliding_win_size,np.nan), trainPredict_scaled[:,0])))
    plt.plot(np.concatenate((np.full(len(dataset_scaled)-forecast_win_size,np.nan), forecast_scaled)))
    plt.show()
    
    plt.rcParams["figure.figsize"] = (16,10)
    plt.title(index_name + " - ZOOM")
    plt.plot(test)
    plt.plot(forecast_scaled)
    plt.show()
    
    return forecast_scaled

def compute_portfolio_variations(forecast):
    size = len(forecast)
    res = []
    for i in range(size-1):
        res.append((forecast[i+1]-forecast[i])/forecast[i])
    return res

def plot_forecast(index_name, dataset, forecast_elems):
    plt.rcParams["figure.figsize"] = (16,10)
    plt.title(index_name)
    plt.plot(dataset)
    plt.plot(np.concatenate((np.full(len(dataset)-forecast_win_size,np.nan), forecast_elems)))
    plt.show()

def save_on_csv(data, csv_name):
    npa = np.array(data)
    npat = npa.transpose()
    df = pd.DataFrame(npat)
    df_new = df.rename(columns={0: 'All_Bonds', 1: 'FTSE_MIB', 2: 'GOLD_SPOT', 3: 'MSCI_EM', 4: 'MSCI_EURO', 5: 'SP_500', 6: 'US_Treasury'})
    df_new.to_csv("../"+csv_name, index=None)
    

if __name__ == "__main__":
    # change working directory to script path
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    indices_epochs = {
        'All_Bonds': ((66,132,), 200),
        'FTSE_MIB': ((132,), 200),
        'GOLD_SPOT': ((66,66,), 350),
        'MSCI_EM': ((66,66,), 200),
        'MSCI_EURO': ((66,132,), 300),
        'SP_500': ((264,), 200),
        'US_Treasury': ((132,132,), 250)
    }
    
    forecasts = []
    dss = sys.argv[1:]
    
    """
    for ds in dss:
        df = pd.read_csv("../" + ds)
        dataset = df.values
        dataset = dataset.astype('float32')
        index_name = ds[:-4]
        hidden_layers = indices_epochs[index_name][0]
        n_epochs = indices_epochs[index_name][1]
        
        scaler = StandardScaler().fit(dataset)   
        dataset_scaled = scaler.transform(dataset)
        f = make_MLP_forecast(index_name, dataset_scaled, hidden_layers, n_epochs)
        f = scaler.inverse_transform(f)
        plot_forecast(index_name, dataset, f)
        forecasts.append(f)

    variations = []
    for f in forecasts:
        variations.append(compute_portfolio_variations(f))

    save_on_csv(forecasts, "forecasts.csv")
    save_on_csv(variations, "variations.csv")
    """
    
    """
    pso = myPSO.PSO(c0=0.25, c1=1.5, c2=2.0, pos_min=-100, pos_max=100, dimensionality=20, num_particles=50, num_neighbors=10)
    pso.run(maxiter=1000)
    """
    
    variations = pd.read_csv("../variations.csv")
    
    port = myPSO.Portfolio(variations, investment=100000, risk_weight=0.5, return__weight=0.5)
    pso = myPSO.PSO(port, c0=0.25, c1=1.5, c2=2, pos_min=0.05, pos_max=1.0, dimensionality=7, num_particles=50, num_neighbors=10)
    pso.run(maxiter=1000)
    
    print('finito')
    
"""
                     R2|Epochs
    'All_Bonds': 92.65%|98
    'FTSE_MIB': 96.65%|27
    'GOLD_SPOT': 98.43%|40
    'MSCI_EM': 94.20%|20
    'MSCI_EURO': 96.12%|44
    'SP_500': 92.40%|37
    'US_Treasury': 98.85%|26
"""
   
   

