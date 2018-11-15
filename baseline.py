from dataParser import *
import pandas as pd
import numpy as np
import config
import worldmap


"""
Baseline uses linear extrapolation between the most recent two points to predict the next future points.
"""
def baseline(worldmap, hurricanes):

    def euclideanDist(pt1, pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

    #Make list of error/predictions made
    aggregateErrors = []
    #For each hurricane in datafile
    for key in hurricanes:
        #Hurricane: is a data structure that contains [[lat, long], [lat, long], [lat, long]] in correct time steps (row, col order)
        hurricane = hurricanes[key]

        #Make error count and make number of predictions count
        totalError = 0
        predictionsMade = 0
        #For each prediction cycle (as see past data + predict future + evaluate future):
        for curIndex in range (0, len(hurricane)-config.FUTURE_VISION):
            if curIndex < 1: continue
            #Get current hurricane's row col and previous one
            curHurricaneRowCol = [worldmap.latToRow(hurricane[curIndex][0]), worldmap.longToCol(hurricane[curIndex][1])]
            prevHurricaneRowCol = [worldmap.latToRow(hurricane[curIndex-1][0]), worldmap.longToCol(hurricane[curIndex-1][1])]
            #Calculate linear difference between current and previous
            diff = [curHurricaneRowCol[0]-prevHurricaneRowCol[0], curHurricaneRowCol[1]-prevHurricaneRowCol[1]]
            for predictIndex in range (1, config.FUTURE_VISION+1):
                newRowCol = [curHurricaneRowCol[0] + predictIndex*diff[0], curHurricaneRowCol[1] + predictIndex*diff[1]]
                #print (hurricane[curIndex+predictIndex][0])
                actual = [worldmap.latToRow(hurricane[curIndex+predictIndex][0]), worldmap.longToCol(hurricane[curIndex+predictIndex][1])]
                error = euclideanDist(newRowCol, actual)
                totalError += error
            predictionsMade += 1 #predictionsMade consists of a set of config.FUTURE_VISION predictions (set of predictions)
        if predictionsMade > 0:
            print ("Hurricane ID " + key + " overall error per prediction is " + str(totalError/predictionsMade))
            aggregateErrors.append(totalError/predictionsMade)
    overallError = sum(aggregateErrors)/len(aggregateErrors)
    print ("Overall error per prediction averaged across all hurricanes: " + str(overallError))
    return overallError

            #Extrapolate line and predict FUTURE_VISION number of data points (or maximum number that doesn't go past hurricane data) (also make sure it's in grid bounds)
            #Calculate error given FUTURE_VISION number of actual data points
            #Update error count and update predictions count
        #Append value of error/prediction made to list
    #Report Average(error/predictions list)





"""
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
"""



worldmap = worldmap.WorldMap()

all_fn = "./data/atlantic.csv"
train_fn = "./train.csv"
valid_fn = "./validate.csv"
test_fn = "./test.csv"
all_df = processCSVFile(all_fn)
train_df = processCSVFile(train_fn)
valid_df = processCSVFile(valid_fn)
test_df = processCSVFile(test_fn)

#TODO use baseline

baseline(worldmap, train_df.getHurricaneLatAndLong())
#baseline(worldmap, valid_df.getHurricaneLatAndLong())
#baseline(worldmap, test_df.getHurricaneLatAndLong())
