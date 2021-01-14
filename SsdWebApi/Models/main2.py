# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:17:20 2021

@author: Federico
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys

from sklearn.preprocessing import StandardScaler
from keras_MLP_lib import MLPUtil
import my_utils as utils
import PSO as myPSO
from sklearn.metrics import mean_squared_error

import json

one_month = 22 # 1 month considering working days only
one_year = one_month*12 # 1 years considering working days only
forecast_win_size = one_year 
sliding_win_size = one_month

def add_ext_var(train_x, ext_data):
    ext_data_time = one_month
    j = int(sliding_win_size / ext_data_time) # index from which start reading ext_data values
    k = ext_data_time # how many times repeat j-th ext_data value
    corrected_train_x = np.zeros((train_x.shape[0], train_x.shape[1]+1))
    for i in range(len(train_x)):
        # add data from external var to the sliding window
        corrected_train_x[i] = np.append(train_x[i], ext_data[j])
        k -= 1
        if k == 0:
            j += 1
            k = ext_data_time
    return corrected_train_x

def load_ext_data(path):
    ext_data = pd.read_csv(path, usecols=[1], names=['value'], header=0).to_numpy()
    scaler = StandardScaler().fit(ext_data)   
    return scaler.transform(ext_data)

def make_MLP_forecast(index_name, dataset_scaled, hidden_layers, n_epochs):
    mlpUtil = MLPUtil(dataset_scaled, sliding_win_size, forecast_win_size, ext_vars_time = one_month)
    
    dataset_scaled = dataset_scaled[-35*sliding_win_size:]
    
    train, test = utils.test_train_split(dataset_scaled, forecast_win_size)

    # sliding window matrices
    train_x, train_y = utils.series_to_supervised(train, sliding_win_size)

    ext_vars_dict = {'DISOC_EURO': [], # add external vars here!
                     'DISOC_USA': [],
                     'GDP_EURO': [],
                     'GDP_USA': [],
                     'INFL_EURO': [],
                     'INFL_USA': [],
                     'PMI_EURO': [],
                     'PMI_USA': []}
    corrected_train_x = train_x
    for v in ext_vars_dict:
        ext_data_scaled = load_ext_data('../ext_vars/' + v + '.csv')
        if (v != 'PMI_EURO'):
            ext_data_scaled = ext_data_scaled[-35:] # 35 are the values that are available in PMI_EURO
        ext_vars_dict[v] = ext_data_scaled
        corrected_train_x = add_ext_var(corrected_train_x, ext_data_scaled)        
    
    # model_t = mlpUtil.train(train_x, train_y, hidden_layers, n_epochs)
    model_t = mlpUtil.train(corrected_train_x, train_y, hidden_layers, n_epochs)
    
    # mlpUtil.train_predict(model_t, train_x)
    trainPredict_scaled = mlpUtil.train_predict(model_t, corrected_train_x)
    
    first_window = train_x[len(train_x)-1] # last window of the training set
    last_ext_vars_index = int(forecast_win_size/one_month)
    # get last last_ext_vars_index values of each ext var
    ext_vars_tuple = ()
    for v in ext_vars_dict:
        ext_var_data = ext_vars_dict[v]
        ext_var_last = ext_var_data[-last_ext_vars_index:]
        ext_vars_tuple += (ext_var_last,)
    
    forecast_scaled = mlpUtil.test_forecast(model_t, first_window, ext_vars_tuple)
    print('Score on test: MSE = {0}'.format(mean_squared_error(test, forecast_scaled)))
    
    plt.rcParams["figure.figsize"] = (16,10)
    plt.title(index_name)
    plt.plot(dataset_scaled)
    plt.plot(np.concatenate((np.full(sliding_win_size, np.nan), trainPredict_scaled[:,0])))
    plt.plot(np.concatenate((np.full(len(dataset_scaled)-forecast_win_size, np.nan), forecast_scaled)))
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
        'All_Bonds': ((60,60,), 200),
        'FTSE_MIB': ((60,60,), 200),
        'GOLD_SPOT': ((30,), 400),
        'MSCI_EM': ((60,60,), 200),
        'MSCI_EURO': ((60,), 300),
        'SP_500': ((60,60,), 200),
        'US_Treasury': ((60,60,), 400)
    }
    forecasts = []
    dss = sys.argv[1:8]

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

    """
    # TEST
    #save_on_csv(forecasts, "forecasts.csv")
    #save_on_csv(variations, "variations.csv")
    
    capital = int(sys.argv[8])
    alpha = float(sys.argv[9])
    
    # TEST
    #variations = pd.read_csv("../variations.csv")
    #port = myPSO.Portfolio(variations.to_numpy(), capital, alpha)

    port = myPSO.Portfolio(np.array(variations).transpose(), capital, alpha)
    
    pso = myPSO.PSO(port, c0=0.25, c1=1.5, c2=2, pos_min=0.05, pos_max=0.7, dimensionality=7, num_particles=20, num_neighbors=6)
    gbest, gfitbest, best_ret, best_risk = pso.run(maxiter=100)
    print("best portfolio: {0} - gfitbest = {1} - best return = {2} - best risk = {3}".format(gbest, gfitbest, best_ret, best_risk))
    
    output = {
        "horizon": 12,
        "S&P_500_INDEX": gbest[5],
        "FTSE_MIB_INDEX": gbest[1],
        "GOLD_SPOT_$_OZ": gbest[2],
        "MSCI_EM": gbest[3],
        "MSCI_EURO": gbest[4],
        "All_Bonds_TR" : gbest[0],
        "U.S._Treasury": gbest[6]
    }
    with open('../portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    output = json.dumps(output)
    
    print('BEST_PORTFOLIO ' + output)
    print('BEST_RETURN {0:.2f}'.format(best_ret))
    print('BEST_RISK {0:.2f}'.format(best_risk))
    """
