from dataParser import *
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("choice", help="Enter 0 for atlantic.csv or 1 for pacific.csv ")
args = parser.parse_args()

dataFile = None
if args.choice == '0':
    fileName = "./data/atlantic.csv"
elif args.choice == '1':
    fileName = "./data/pacific.csv"
else:
    print("Invalid input, exiting...")
    sys.exit()
dataFile = processCSVFile(fileName)

# Parse hurricane ID to separate hurricanes
H_map, H_map_order = dataFile.getHurricaneDict()
latitude, longitude = dataFile.getLatAndLong()

# Plot each hurricane
num = len(H_map)
print("Displaying " + str(num) + " number of hurricane paths")
for key in H_map:
    points = H_map[key]
    x = longitude[points]
    y = latitude[points]

    plt.xlim(-120, 0)
    plt.ylim(0, 100)
    plt.ylabel("latitude")
    plt.xlabel("longitude")
    # Plots all coords blue, then first landing coords green, then US coastliner red
    plt.plot(x, y, 'bo')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
