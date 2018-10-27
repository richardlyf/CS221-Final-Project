from dataParser import *
import numpy as np
import pandas as pd

class worldMap:
    '''
    Initializes the world map to grid of zeros
    lat_span, long_span indicates the size of the world
    '''
    def __init__(self, lat_span, long_span):
        self.lat_span = lat_span
        self.long_span = long_span
        self.worldGrid = np.zeros((lat_span, long_span))
        self.avgWind = np.zeros((lat_span, long_span))

        # Below are values assumed for baseline. Real values can be read from dataset

        # For baseline the average wind is assumed to be 70 mph
        self.avgWind += 1
        # For baseline the radius of Hurricane is assumed to be 3 degrees
        self.radius = 3

    '''
    For each Hurrican that pass through a world cell, add 1 to that cell
    Also keep track of average wind for the particular cell
    Compute the risk by calculating the percentage of Hurricanes that pass through the cell
    Risk = % * average_wind
    '''
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

    '''
    Compare training and validation data and calculate the risk difference between each world cell
    '''
    def evaluation(self, train_df, valid_df):
        risk_train = self.count(train_df)
        risk_valid = self.count(valid_df)
        train_valid = np.sum(abs(risk_train - risk_valid))
        train_train = np.sum(abs(risk_train - risk_train))
        print("Train")
        print(len(np.nonzero(risk_train)[0]))
        print("valid")
        print(len(np.nonzero(risk_valid)[0]))
        # Baseline
        #print("Baseline: diff training and training: " + str(train_train))
        #print("Baseline: diff training and validation/test: " + str(train_valid))

        # Oracle
        #print("Oracle: diff training and training: " + str(train_train))
        #print("Oracle: diff training and validation/test: " + str(train_valid))


wm = worldMap(1000, 1500)

all_fn = "./data/atlantic.csv"
train_fn = "./train.csv"
valid_fn = "./validate.csv"
test_fn = "./test.csv"
all_df = processCSVFile(all_fn)
train_df = processCSVFile(train_fn)
valid_df = processCSVFile(valid_fn)
test_df = processCSVFile(test_fn)

#wm.evaluation(all_df, train_df)

wm.evaluation(train_df, valid_df)
wm.evaluation(train_df, test_df)




