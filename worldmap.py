import numpy as np
import config

class WorldMap:
    '''
    Initializes the world map to grid of zeros
    lat_span, long_span indicates the size of the world
    '''
    def __init__(self):
        self.lat_span = config.LATITUDE_RANGE*config.GRID_SIZE_SCALE
        self.long_span = config.LONGITUDE_RANGE*config.GRID_SIZE_SCALE
        self.worldGrid = np.zeros((self.lat_span, self.long_span))

    '''
    Convert latitude to row in grid representation
    '''
    def latToRow(self, latitude):
        val = latitude * config.GRID_SIZE_SCALE
        assert(val >= 0 and val < self.lat_span)
        return val

    '''
    Convert longitude to col in grid representation
    '''
    def longToCol(self, longitude):
        val = -longitude * config.GRID_SIZE_SCALE #Negative from longitude being negative in our area of concern
        assert(val >= 0 and val < self.long_span)
        return val

    '''
    Resets the grid of all values
    '''
    def zeroGrid(self):
        self.worldGrid = np.zeros(self.lat_span, self.long_span)


    '''
    Marks value at row, col in the grid
    '''
    def setValueAt(self, row, col, value):
        assert (type(value) == int)
        assert (row >= 0 and row < self.lat_span)
        assert (col >= 0 and col < self.long_span)
        self.worldGrid[row, col] = value

    '''
    Gets value at row, col, in the grid
    '''
    def getValueAt(self, row, col):
        assert (row >= 0 and row < self.lat_span)
        assert (col >= 0 and col < self.long_span)
        return self.worldGrid[row, col]











    '''
    For each Hurricane that pass through a world cell, add 1 to that cell
    Also keep track of average wind for the particular cell
    Compute the risk by calculating the percentage of Hurricanes that pass through the cell
    Risk = % * average_wind
    '''
    """
    def count(self, df):
        latitude, longitude = df.getLatAndLong()
        longitude = longitude * -1
        numHurricane = len(latitude)
        print("building grid...")
        for i in range(numHurricane):
            lat = int(latitude[i] * 10)
            lon = int(longitude[i] * 10)
            self.worldGrid[lat - self.radius : lat + self.radius, lon - self.radius : lon + self.radius] += 1

        print("Calculating risk...")
        riskGrid = np.zeros((self.lat_span, self.long_span))
        for lat in range(self.lat_span):
            for lon in range(self.long_span):
                riskGrid[lat, lon] = self.avgWind[lat, lon] * self.worldGrid[lat, lon] / numHurricane

        self.worldGrid = np.zeros((self.lat_span, self.long_span))
        return riskGrid
    """
