## hello there
## this file is a python script that will detect anomalies in a coninues data stream!

import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def generate_data(size = 1000):
    normal_data = np.random.normal(0, 1, size)
    anomalies = np.random.uniform(10, 15, size//100)
    data_stream = np.concatenate((normal_data, anomalies))
    np.random.shuffle(data_stream)
    return data_stream

def anomalies_detector(data_stream):
    isolate_forest = IsolationForest(contamination = 0.1)
    data = np.array(data_stream).reshape(-1,1)
    predictions = isolate_forest.fit_predict(data)
    return predictions

def simulate(stream):
    predictions = anomalies_detector(stream)
    for i,point in enumerate(stream):
        print(f"Data: {point}, Anomaly: {predictions[i] == -1}")

def current_plot(stream):
    plt.ion()
    fig, ax = plt.subplots()
    data = []

    for i,point in enumerate(stream):
        data.append(point)
        ax.clear()
        ax.plot(data, label='Data Stream', color='blue')

        if predictions[i] == -1:
            ax.scatter(len(data) -1 , point, color='red', label='Anamoly')
        plt.pause(0.08)
    plt.show()

def main():
    stream = generate_data()

    print("starting simulation...")
    simulate(stream)

    print("starting visualization...")
    current_plot(stream)

if __name__ == "__main__":
    main()
