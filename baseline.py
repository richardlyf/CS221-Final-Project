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

            #Predict the next few timesteps and sum up the errors
            for predictIndex in range (1, config.FUTURE_VISION+1):
                newRowCol = [curHurricaneRowCol[0] + predictIndex*diff[0], curHurricaneRowCol[1] + predictIndex*diff[1]]
                actual = [worldmap.latToRow(hurricane[curIndex+predictIndex][0]), worldmap.longToCol(hurricane[curIndex+predictIndex][1])]
                error = euclideanDist(newRowCol, actual)
                totalError += error
            predictionsMade += 1 #predictionsMade consists of a set of config.FUTURE_VISION predictions (set of predictions)

        #Print out and add this hurricane's error per prediction to our list
        if predictionsMade > 0:
            print ("Hurricane ID " + key + " overall error per prediction is " + str(totalError/predictionsMade))
            aggregateErrors.append(totalError/predictionsMade)

    #Average out errors across all hurricanes
    overallError = sum(aggregateErrors) / len(aggregateErrors) / config.FUTURE_VISION
    print ("Overall error per prediction averaged across " + str(len(aggregateErrors)) + " hurricanes: " + str(overallError))
    return overallError




#Initialize world map
worldmap = worldmap.WorldMap()


#Prepare data
all_fn = "./data/atlantic.csv"
train_fn = "./train.csv"
valid_fn = "./validate.csv"
test_fn = "./test.csv"
all_df = processCSVFile(all_fn)
train_df = processCSVFile(train_fn)
valid_df = processCSVFile(valid_fn)
test_df = processCSVFile(test_fn)


#Which data for the baseline to run on

#baseline(worldmap, train_df.getHurricaneLatAndLong())
baseline(worldmap, valid_df.getHurricaneLatAndLong())
#baseline(worldmap, test_df.getHurricaneLatAndLong())
