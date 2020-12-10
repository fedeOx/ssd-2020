# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:06:42 2020

@author: Federico
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys

from sklearn.preprocessing import StandardScaler
import MLP_lib as myMLP
import my_utils as utils

forecast_win_size = 240 # 1 years considering working days only
sliding_win_size = 22*2 # 2 month considering working days only

def make_MLP_forecast(index_name, dataset_scaled, n_hidden, n_epochs):    
    train, test = utils.test_train_split(dataset_scaled, forecast_win_size)

    # sliding window matrices
    train_x, train_y = utils.compute_windows(train, sliding_win_size)
    test_x, test_y = utils.compute_windows(test, sliding_win_size)
    mlp_trained = myMLP.train(train_x, train_y, test_x, test_y, n_hidden, n_epochs)
    
    trainPredict_scaled, testForecast_scaled = myMLP.predict(mlp_trained, train_x, test_x)
    
    forecast_elems_scaled = myMLP.forecast(mlp_trained, train_x, forecast_win_size)
      
    plt.rcParams["figure.figsize"] = (10,8)
    plt.title(index_name)
    plt.plot(dataset_scaled)
    plt.plot(np.concatenate((np.full(sliding_win_size,np.nan), trainPredict_scaled)))
    plt.plot(np.concatenate((np.full(sliding_win_size+len(train),np.nan), testForecast_scaled)))
    plt.show()
    
    return forecast_elems_scaled

def compute_portfolio_variations(forecast):
    size = len(forecast)
    res = []
    for i in range(size-1):
        res.append((forecast[i+1]-forecast[i])/forecast[i])
    return res

def plot_forecast(index_name, dataset, forecast_elems):
    plt.rcParams["figure.figsize"] = (10,8)
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
        'All_Bonds': ((22,22), 98),
        'FTSE_MIB': ((100,100), 27),
        'GOLD_SPOT': ((22,22), 80),
        'MSCI_EM': ((22,22), 20),
        'MSCI_EURO': ((100,100), 44),
        'SP_500': ((100,100), 37),
        'US_Treasury': ((100,100), 26)
    }
    
    forecasts = []
    dss = sys.argv[1:]
    
    for ds in dss:
        df = pd.read_csv("../" + ds)
        dataset = df.values
        dataset = dataset.astype('float32')
        index_name = ds[:-4]
        
        scaler = StandardScaler().fit(dataset)   
        dataset_scaled = scaler.transform(dataset)
        f = make_MLP_forecast(index_name, dataset_scaled,
                              n_hidden=indices_epochs[index_name][0], n_epochs=indices_epochs[index_name][1])
        f = scaler.inverse_transform(f)
        plot_forecast(index_name, dataset, f)
        forecasts.append(f)

    variations = []
    for f in forecasts:
        variations.append(compute_portfolio_variations(f))

    save_on_csv(forecasts, "forecasts.csv")
    save_on_csv(variations, "variations.csv")

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
   
   

