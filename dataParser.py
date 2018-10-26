import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
plt.style.use('seaborn-whitegrid')

'''
Returns the x, y coodinates that form the US coastline
'''
def getUSCoastline():
    topRight = [-63.5, 47.1]
    botLeft = [-83.7, 28.7]
    diff_y = topRight[1] - botLeft[1]
    diff_x = topRight[0] - botLeft[0]
    dy = diff_y / diff_x
    intercept = topRight[1] - dy * topRight[0]

    # US coast line
    x = np.arange(botLeft[0], topRight[0], 0.1)
    y = dy * x + intercept
    return x, y

'''
Helper class that reads CSV files
'''
class processCSVFile:
    def __init__(self, labels, fileName):
        self.labels = labels
        self.file_reader = pd.read_csv(fileName)

    '''
    Takes in a list of labels and return a list with lists of data corresponding to the labels
    '''
    def getRawData(self, labels):
        result = []
        for label in labels:
            result.append(self.file_reader[label])
        return result

    '''
    Return the latitude and longitude in gloal coordinates
    '''
    def getLatAndLong(self):
        plotData = self.getRawData(['Latitude', 'Longitude'])
        latitude = self.stripDirection(plotData[0])
        #From GPS coord to actual coord. This only applies to Atlantic data, because W is equivalent to -1
        longitude = self.stripDirection(plotData[1]) * -1
        return latitude, longitude

    '''
    Returns the coordinate for when the hurricane first landed (time = 0)
    '''
    def getLandingLatAndLong(self):
        latitude, longitude = self.getLatAndLong()
        time = self.getRawData(['Time'])[0]
        t0_index = np.where(time == 0)[0]
        t0_lat = np.take(latitude, t0_index)
        t0_long = np.take(longitude, t0_index)
        return t0_lat, t0_long

    '''
    Strips the direction W N S E from longitude and latitude data
    '''
    def stripDirection(self, dataList):
        return np.asarray([float(data[:-1]) for data in dataList])

if __name__ == "__main__":
    fileName = "./data/atlantic.csv"
    labels = ['ID', 'Name', 'Date', 'Time', 'Latitude', 'Longitude']
    dataFile = processCSVFile(labels, fileName)

    latitude, longitude = dataFile.getLatAndLong()
    t0_lat, t0_long = dataFile.getLandingLatAndLong()
    x, y = getUSCoastline()

    plt.ylabel("latitude")
    plt.xlabel("longitude")
    # Plots all coords blue, then first landing coords green, then US coastliner red
    plt.plot(longitude, latitude, 'bo', t0_long, t0_lat, 'g^', x, y, 'r')
    plt.show()
