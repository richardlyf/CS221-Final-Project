'''
The program is supposed to take a CSV file and plit it into Train, Validation, and Test set.
The output will be stored as CSV files in the same directory
'''
from dataParser import *
import numpy as np
import pandas as pd

fileName = "./data/atlantic.csv"
dataFile = processCSVFile(fileName)
partition = [0.6, 0.2, 0.2]

# Calculate partition size
H_map = dataFile.getHurricaneDict()
dataSize = len(H_map)
trainSize = int(dataSize * partition[0])
validationSize = int(dataSize * partition[1])
testSize = dataSize - trainSize - validationSize

# Get all hurricane IDs
H_ids = np.asarray(H_map.keys())
# Get all rows of data
rows = np.asarray(dataFile.getRow(0, dataSize))

# Allocate data by partition
train_ids = H_ids[:trainSize]
#Randomize for validation and test
H_ids = H_ids[trainSize:]
np.random.shuffle(H_ids)
validate_ids = H_ids[: validationSize]
test_ids = H_ids[validationSize:]





# Save data to CSV
headerNames = ','.join(dataFile.getNames())
np.savetxt('train.csv', train, fmt='%s', delimiter=',', header=headerNames, comments='')
np.savetxt('validate.csv', validate, fmt='%s', delimiter=',', header=headerNames, comments='')
np.savetxt('test.csv', test, fmt='%s', delimiter=',', header=headerNames, comments='')
