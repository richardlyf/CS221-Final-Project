'''
A basic mle model that accepts traning data and learns the parameters of the Bayesian network
'''
from dataParser import *
import pandas as pd
import numpy as np
import config
import worldmap
from collections import defaultdict
import random

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
            # Convert hurricane[n] coord to worldmap coord, and store each point coord in a tuple for N_PAST_POINTS
            prior_key = tuple([(worldmap.latToRow(hurricane[n][0]), worldmap.longToCol(hurricane[n][1])) for n in range(i - config.PAST_VISION, i)])
            curr_key = (worldmap.latToRow(hurricane[i][0]), worldmap.longToCol(hurricane[i][1]))
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
        p_table[prior_key] = {key : value / total for key, value in evidence_counts.items()}


    return p_table

'''
Takes in a probability table to modify
Takes in the prior and the next_point
Adds a gaussian distribution to the probability table centered at next_point given the prior
The config file specifies the standard diviation estimate(given range/threshold) and mean of the gaussian
'''
def gaussianLaplacian(p_table, curr_key, prior_key):
    pass



'''
Takes in a probability table and test data and computes the predicted data points.
Evaluates the predicted data points and prints out error.
'''
def predictAndEval(worldMap, dataFile, p_table):
    hurricanes = dataFile.getHurricaneLatAndLong()
    hurricane_errors = []
    total_num_predictions = 0
    # For each hurricane
    for key in hurricanes:
        hurricane = hurricanes[key]
        total_error = 0
        prediction_count = (len(hurricane) - config.FUTURE_VISION - config.PAST_VISION) * config.PARTICLE_AMOUNT
        # Look at n PAST_VISION points and make a prediction about the next FUTURE_VISION point(s)
        for i in range(config.PAST_VISION, len(hurricane) - config.FUTURE_VISION):
            particle_sum_error = 0

            # For each particle at this time step
            for particle in range(config.PARTICLE_AMOUNT):
                #Simulate particles as hurricanes to more robustly evaluate Bayes net
                ######################################################
                skip = False
                ######################################################

                predicted_points = []

                # Form the prior
                prior_key = tuple([(worldmap.latToRow(hurricane[n][0]), worldmap.longToCol(hurricane[n][1])) for n in range(i - config.PAST_VISION, i)])
                # Predict FUTURE_VISION points given a prior
                for shift in range(config.FUTURE_VISION):
                    # Pick next potential point from the probability distribution
                    evidence = p_table[prior_key]

                    #########TEMP SOLUTION TO EMPTY WEIGHT###########
                    if len(evidence) == 0:
                        prediction_count -= 1 #TODO is this -= 1 or actually just 0 now?  with -1 doesn't make sense as
                                                #you're discarding the rest of the points which is more than 1 discarded
                        #prediction_count = 0
                        skip = True
                        break;
                    #################################################

                    prediction = weightedRandomChoice(evidence)
                    predicted_points.append(prediction)
                    # Update prior
                    _new_prior = list(prior_key)[1:]
                    _new_prior.append(prediction)
                    prior_key = tuple(_new_prior)

                #################################################
                if skip:
                    continue
                    #break
                #################################################

                # Use the list of predicted_points given prior to calculate particle error
                target_points = [(worldmap.latToRow(hurricane[t][0]), worldmap.longToCol(hurricane[t][1])) for t in range(i, i + config.FUTURE_VISION)]
                particle_sum_error += calculateError(predicted_points, target_points)

            #Add the particles' average error to total error
            total_error += particle_sum_error

        #####
        if prediction_count == 0:
            continue
        #####

        total_num_predictions += prediction_count
        avg_hurricane_error = total_error / prediction_count
        hurricane_errors.append(avg_hurricane_error)
        print ("Hurricane ID " + key + " overall error per prediction is " + str(avg_hurricane_error))

    total_avg_error = sum(hurricane_errors) / len(hurricane_errors) / config.FUTURE_VISION
    print ("Overall error per prediction averaged across " + str(len(hurricane_errors)) + " hurricanes and "
            + str(total_num_predictions) + " predictions: " + str(total_avg_error))

'''
Takes in a list of predicted values and a list of target values
Returns an error as the sum of L2 distances
'''
def calculateError(prediction, target):
    total = 0
    for i in range(len(prediction)):
        total += euclideanDist(prediction[i], target[i])
    return total

'''
Accepts pt1 and pt2 as coordinate tuples
Returns the L2 distance between two coordinates
'''
def euclideanDist(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5


'''
Given a dictionary of the form element -> weight, selects an element
randomly based on distribution proportional to the weights. Weights can sum
up to be more than 1.
'''
def weightedRandomChoice(weightDict):
    weights = []
    elems = []
    for elem in sorted(weightDict):
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    chosenIndex = None
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]
    raise Exception('Should not reach here')



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
predictAndEval(worldmap, valid_df, table)
