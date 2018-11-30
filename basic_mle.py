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
import scipy
import scipy.stats as st

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

    #Dictionary storing whether we already added Laplacian or not for (prior, current) as key, with
    #a value of 1 if already added Laplace or it's not in the dictionary if not
    laplaceAlreadyAdded = defaultdict(int)

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
            if (prior_key, curr_key) not in laplaceAlreadyAdded:
                gaussianLaplacian(p_table, prior_key, curr_key)
                #Mark this as already added Laplacian to avoid repeat adding #TODO check
                laplaceAlreadyAdded[(prior_key, curr_key)] = 1

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
def gaussianLaplacian(p_table, prior_key, curr_key = None): #TODO check
    #For all values(x, y) in our dictionary within LAPLACE_RADIUS
    for deltaRow in range (-config.LAPLACE_RADIUS, config.LAPLACE_RADIUS):
        for deltaCol in range (-config.LAPLACE_RADIUS, config.LAPLACE_RADIUS):
            #Ensure circular area of points that we're going to add Laplacian to
            if (euclideanDist((deltaRow, deltaCol), (0, 0)) > config.LAPLACE_RADIUS):
                continue

            new_key = (curr_key[0] + deltaRow, curr_key[1] + deltaCol)

            #Gaussian value as a 0 centered, with 1 stdev as LAPLACE_RADIUS.  We're taking the probability
            #of our deviation given by the Euclidean distance of the deltas with respect to the origin
            #We then multiply the entire Gaussian by a scalar defined by LAPLACE_LAMBDA
            laplaceValue = scipy.stats.norm(0, config.LAPLACE_RADIUS).pdf(euclideanDist((deltaRow, deltaCol), (0, 0))) * config.LAPLACE_LAMBDA

            if new_key in p_table[prior_key]:
                p_table[prior_key][new_key] += laplaceValue
            else:
                p_table[prior_key][new_key] = laplaceValue


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

                predicted_points = []

                # Form the prior
                prior_key = tuple([(worldmap.latToRow(hurricane[n][0]), worldmap.longToCol(hurricane[n][1])) for n in range(i - config.PAST_VISION, i)])
                # Predict FUTURE_VISION points given a prior
                for shift in range(config.FUTURE_VISION):
                    # Pick next potential point from the probability distribution
                    evidence = p_table[prior_key]

                    #If haven't seen before, use Laplace to crudely estimate it
                    if len(evidence) == 0:

                        #We have nothing for the p_table entry, so we should add
                        #Laplace right away to calculate next point.  Default is a linear
                        #predictor.

                        #Linear predictor for extrapolating from at least 2 prior key points
                        assert(len(prior_key) >= 2)
                        curr_key = (2*prior_key[-1][0] - prior_key[-2][0], 2*prior_key[-1][1] - prior_key[-2][1])
                        gaussianLaplacian(p_table, prior_key, curr_key)
                        #Normalize Laplacian
                        evidence_counts = p_table[prior_key]
                        total = sum(evidence_counts.values())
                        p_table[prior_key] = {key : value / total for key, value in evidence_counts.items()}


                    prediction = weightedRandomChoice(evidence)
                    predicted_points.append(prediction)
                    # Update prior
                    _new_prior = list(prior_key)[1:]
                    _new_prior.append(prediction)
                    prior_key = tuple(_new_prior)

                # Use the list of predicted_points given prior to calculate particle error
                target_points = [(worldmap.latToRow(hurricane[t][0]), worldmap.longToCol(hurricane[t][1])) for t in range(i, i + config.FUTURE_VISION)]
                particle_sum_error += calculateError(predicted_points, target_points)

            #Add the particles' average error to total error
            total_error += particle_sum_error

        if prediction_count == 0:
            continue

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
