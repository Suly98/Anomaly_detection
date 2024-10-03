## hello there
## this file is a python script that will detect anomalies in a coninues data stream!

import numpy as np

def generate_data(size = 1000):
    normal_data = np.random.normal(0, 1, size)
    anomalies = np.random.uniform(10, 15, size//100)
    data_stream = np.concatenate(normal_data, anomalies)
    np.random.shuffle(data_stream)

    return data_stream

from sklearn.ensemble import IsolationForest

def anomalies_detector(data_stream):
    isolate_forest = IsolationForest(contamination = 0.01)
    data = np.array(data_stream).reshape(-1,1)
    predictions = isolate_forest.fit_predict(data)
    return predictions

import time

def simulate():
    stream = generate_data()
    for point in stream:
        predictions = anomalies_detector([point])
        print(f"Data: {point}, Anomaly: {predictions[0] == -1}")
        time.sleep(0.1)


import matplotlib.pyplot as plt

def current_plot(stream):
    