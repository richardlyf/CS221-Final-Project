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
dataSize = dataFile.getDataSize()
trainSize = int(dataSize * partition[0])
validationSize = int(dataSize * partition[1])
testSize = dataSize - trainSize - validationSize

# Randomize data
rows = np.asarray(dataFile.getRow(0, dataSize))
np.random.shuffle(rows)

# Allocate data by partition
train = rows[:trainSize]
validate = rows[trainSize : trainSize + validationSize]
test = rows[trainSize + validationSize:]

# Save data to CSV
headerNames = ','.join(dataFile.getNames())
np.savetxt('train.csv', train, fmt='%s', delimiter=',', header=headerNames, comments='')
np.savetxt('validate.csv', validate, fmt='%s', delimiter=',', header=headerNames, comments='')
np.savetxt('test.csv', test, fmt='%s', delimiter=',', header=headerNames, comments='')
