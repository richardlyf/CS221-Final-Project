'''
A basic mle model that accepts traning data and learns the parameters of the Bayesian network
'''
from dataParser import *
import pandas as pd
import numpy as np
import config
import worldmap
from collections import defaultdict

'''
Takes in the training data of each hurricane and computes returns the probability table
@param hurricanes is a map with ID as keys and a list of (lat, long)
'''
def train(worldmap, dataFile):
    '''
    The probability table is a dictionary of dictionaries. The outer most layer uses a tuple of PAST_VISION points as keys,
    and its value is a ditionary mapping the next_point(key) to the count
    i.e. We have data [1, 2, 3, 4] where [1, 2, 3] is the seen data and [4] is the next_point.
    When we see this combination we need to increment count:
    p_table:
    prior_key    (1, 2, 3):
    evidence_counts     (4) : count + 1
             curr_key-> (n) : count
    '''
    hurricanes = dataFile.getHurricaneLatAndLong()
    p_table = defaultdict(dict)
    # Count
    for key in hurricanes:
        hurricane = hurricanes[key]
        # Look at first n PAST_VISION points, add 1 to a dict where its key is (N_PAST_POINTS, next_point)
        # i represents the index of next_point
        for i in range(config.PAST_VISION, len(hurricane)):
            # Increment by 1 along the hurricane and look at the next n PAST_VISION points until there's no more next point
            prior_key = tuple([hurricane[n] for n in range(i - config.PAST_VISION, i)])
            curr_key = hurricane[i]
            evidence_counts = p_table[prior_key]
            if curr_key in evidence_counts:
                evidence_counts[curr_key] += 1
            else:
                evidence_counts[curr_key] = 1

    # Normalize
    prior_keys = p_table.keys()
    for prior_key in prior_keys:
        evidence_counts = p_table[prior_key]
        total = sum(evidence_counts.values())
        evidence_counts = {key : value / total for key, value in evidence_counts.items()}

    return p_table

'''
Takes in a probability table and test data and computes the predicted data points.
'''
def predict(worldMap, dataFile, p_table):
    hurricanes = dataFile.getHurricaneLatAndLong()
    for



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

table = train(worldmap, train_df)
evidence = table[((22.7, -74.5), (23.7, -74.8), (24.6, -74.9))]
keys = list(evidence.keys())
print(evidence[(25.5, -75.0)])
