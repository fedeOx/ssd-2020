# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:43:42 2020

@author: Federico
"""
import numpy as np

def compute_windows(nparray, npast=1):
    dataX, dataY = [], [] # window and value
    for i in range(len(nparray)-npast-1):
        a = nparray[i:(i+npast), 0]
        dataX.append(a)
        dataY.append(nparray[i + npast, 0])
    return np.array(dataX), np.array(dataY)

def test_train_split(dataset, fw_size):
    train = dataset[:-fw_size]
    test = dataset[-fw_size:]
    return train, test