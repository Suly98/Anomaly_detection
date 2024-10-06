## hello there
## this file is a python script that will detect anomalies in a coninues data stream!

import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import time

# func to generate data and outliers
def generate_data(size = 1000):
    normal_data = np.random.normal(0, 1, size) #generating normal data
    anomalies = np.random.uniform(10, 15, size//100) #generatin anomaly data
    data_stream = np.concatenate((normal_data, anomalies)) # mixing them together
    np.random.shuffle(data_stream) # shuffling normal data with anomaly data
    return data_stream

# func to detect anomalies
def anomalies_detector(data_stream):
    isolate_forest = IsolationForest(contamination = 0.1) #starting the algorithm with %10 prediction in having anomalies
    data = np.array(data_stream).reshape(-1,1) #reshaping data so algorithm defines outliers
    predictions = isolate_forest.fit_predict(data) #making the algorithm work
    return predictions

# func to check if data is outliers or inliers and printing them to mimc real time detection.
def simulate(stream):
    predictions = anomalies_detector(stream)
    for i,point in enumerate(stream):
        print(f"Data: {point}, Anomaly: {predictions[i] == -1}")
        time.sleep(0.001)

# func to visualize the data in real time.
def current_plot(stream):
    predictions = anomalies_detector(stream) 
    plt.ion() #adding every update while the script is running
    fig, ax = plt.subplots() #create figure with axes
    data = []

    for i,point in enumerate(stream):
        data.append(point)
        ax.clear() #clear the axes!
        ax.plot(data, label='Data Stream', color='blue') #gives a blue line for the data

        if predictions[i] == -1: # if data is anomaly put it in red.
            ax.scatter(len(data) -1 , point, color='red', label='Anomaly') 
        plt.pause(0.0001) #not really necessary but to make it in real time.

    ax.legend() # displaying the label in reference to the blue line
    plt.show() #showing figure

# main function to call all function and run the project.
def main():
    stream = generate_data()

    print("starting simulation...")
    simulate(stream)

    print("starting visualization...")
    current_plot(stream)

# this will help other developers when they use the script to deploy
# and it is to control the flow of the execution. 
if __name__ == "__main__":
    main()
