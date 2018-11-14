'''
The program handles reading CSV file contents. Running the file directly will plot the data in the CSV file
'''
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
    def __init__(self, fileName):
        self.file_reader = pd.read_csv(fileName)

    '''
    Returns a list of column names
    '''
    def getNames(self):
        return list(self.file_reader)

    '''
    Returns the length of column vector
    '''
    def getDataSize(self):
        return len(self.file_reader['ID'])

    '''
    Returns the entire specified row or rows in the CSV file
    rowL and rowH determines the lower and upper bound of the rows, rowL inclusive rowH exclusive
    If only rowL is entered, only rowL is returned
    '''
    def getRow(self, rowL, rowH = None):
        if rowH == None:
            return self.file_reader[rowL : rowL + 1]
        elif rowH < rowL:
            print("Bounding error! rowH should be at least equal to rowL")
            return None
        return self.file_reader[rowL : rowH]

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
        landing = self.getRawData(['Event'])[0]
        t0_index = np.where(landing == ' L')[0]
        t0_lat = np.take(latitude, t0_index)
        t0_long = np.take(longitude, t0_index)
        return t0_lat, t0_long

    '''
    Strips the direction W N S E from longitude and latitude data
    '''
    def stripDirection(self, dataList):
        return np.asarray([float(data[:-1]) for data in dataList])

    '''
    @H_map
    Returns a dictionary where the keys are hurricanes IDs and the value is
    a list of row numbers containing a hurricane's data
    @H_map_order
    The map above does not keep track of the order of hurricanes. This
    is a list of hurricane IDs appearing in order of time
    '''
    def getHurricaneDict(self):
        IDs = self.getRawData(['ID'])[0]
        H_map = defaultdict(list)
        prev_id = IDs[0]
        H_map_order = [prev_id]
        for i in range(len(IDs)):
            H_map[IDs[i]].append(i)
            if prev_id != IDs[i]:
                H_map_order.append(IDs[i])
                prev_id = IDs[i]
        assert len(H_map.keys()) == len(H_map_order)
        return H_map, H_map_order

    '''
    Similar to the method above.
    Returns a dictionary where the keys are hurricanes IDs and the value is
    a list of (latitude, longitude) tuples of the hurricane's position in order as they occur
    '''
    def getHurricaneLongAndLat(self):
        IDs = self.getRawData(['ID'])[0]
        H_location_map = defaultdict(list)
        latitude, longitude = self.getLatAndLong()
        for i in range(len(IDs)):
            H_map[IDs[i]].append((latitude[i], longitude[i]))
        return H_location_map


if __name__ == "__main__":
    fileName = "./data/atlantic.csv"
    dataFile = processCSVFile(fileName)

    latitude, longitude = dataFile.getLatAndLong()
    t0_lat, t0_long = dataFile.getLandingLatAndLong()
    x, y = getUSCoastline()

    plt.ylabel("latitude")
    plt.xlabel("longitude")
    # Plots all coords blue, then first landing coords green, then US coastliner red
    plt.plot(longitude, latitude, 'bo', t0_long, t0_lat, 'g^', x, y, 'r')
    plt.show()
