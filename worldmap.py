import numpy as np
import config

class WorldMap:
    '''
    Initializes the world map to grid of zeros
    lat_span, long_span indicates the size of the world
    '''
    def __init__(self):
        self.lat_span = int(config.LATITUDE_RANGE/config.GRID_SIZE_SCALE)
        self.long_span = int(config.LONGITUDE_RANGE/config.GRID_SIZE_SCALE)
        self.worldGrid = np.zeros((self.lat_span, self.long_span))

    '''
    Convert latitude to row in grid representation
    '''
    def latToRow(self, latitude):
        val = int(latitude/config.GRID_SIZE_SCALE)
        assert(val >= 0 and val < self.lat_span)
        return val

    '''
    Convert longitude to col in grid representation
    '''
    def longToCol(self, longitude):
        val = int(-longitude/config.GRID_SIZE_SCALE) #Negative from longitude being negative in our area of concern
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
