'''
The program is supposed to take a CSV file and plit it into Train, Validation, and Test set.
The output will be stored as CSV files in the same directory
'''
from dataParser import *
import numpy as np
import pandas as pd
import sys

choice = input("Are you sure you want to repartition data? Doing so will override the past partitioned files. (y/n)")
if choice != 'y':
    print("Exiting...")
    sys.exit()

fileName = "./data/atlantic.csv"
dataFile = processCSVFile(fileName)
partition = [0.6, 0.2, 0.2]

# Calculate partition size
H_map, H_map_order = dataFile.getHurricaneDict()
HSize = len(H_map)
trainSize = int(HSize * partition[0])
validationSize = int(HSize * partition[1])
testSize = HSize - trainSize - validationSize

print("Train data number hurricanes: " + str(trainSize))
print("Validation data number hurricanes: " + str(validationSize))
print("Test data number hurricanes: " + str(testSize))

# Get all rows of data
rows = np.asarray(dataFile.getRow(0, dataFile.getDataSize()))

# Allocate data by partition
# Group the hurricane IDs in partitioned sets
train_ids = H_map_order[:trainSize]
#Randomize for validation and test
H_map_order = H_map_order[trainSize:]
np.random.shuffle(H_map_order)
validate_ids = H_map_order[: validationSize]
test_ids = H_map_order[validationSize:]

# Get the row indices that correspond to each ID in each set
train_row_idx = [point for ID in train_ids for point in H_map[ID]]
valid_row_idx = [point for ID in validate_ids for point in H_map[ID]]
test_row_idx = [point for ID in test_ids for point in H_map[ID]]

train = rows[train_row_idx]
validate = rows[valid_row_idx]
test = rows[test_row_idx]

print("Length of training data: " + str(len(train_row_idx)))
print("Length of validation data: " + str(len(valid_row_idx)))
print("Length of test data: " + str(len(test_row_idx)))

# Save data to CSV
#headerNames = ','.join(dataFile.getNames())
#np.savetxt('train.csv', train, fmt='%s', delimiter=',', header=headerNames, comments='')
#np.savetxt('validate.csv', validate, fmt='%s', delimiter=',', header=headerNames, comments='')
#np.savetxt('test.csv', test, fmt='%s', delimiter=',', header=headerNames, comments='')
