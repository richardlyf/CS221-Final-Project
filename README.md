# CS221-Final-Project

Hurricane trajectory prediction using Bayesian network

CodaLab: https://worksheets.codalab.org/worksheets/0xd64b3fc9cb104e5fba332cdd8246f124/ 

## Files:

#### Programs:

baseline.py: runs the baseline linear extrapolator

config.py: configurations and hyperparameters

dataParser.py: handles reading and plotting CSV file contents. Running this file directly displays all data points of the atlantic data set on the map.

mle.py: body of the MLE and the MLE+Laplace

partitionData.py: splits data set into Train, Validation, and Test.

plotData.py: animates the entire atlantic dataset on the map 

worldmap.py: infrastructure of the world map

#### Data:

The following data files are partitioned from `atlantic.csv` by `partitionData.py`

> test.csv

> validate.csv

> test.csv

The data folder contains the original data files.


## Setting up the environment

This project uses Python3

Install virtualenv and create new environment:

> sudo pip install virtualenv
> virtualenv -p python3 .env
> source .env/bin/activate

Install dependencies:

> pip install -r requirements.txt


## Code Instructions:

Split data: run `python partitionData.py`


Configurations and Hyperparameters: open `config.py` to edit parameters


Baseline: run `python baseline.py`


MLE: run `python mle.py`


dataParser: run `python dataParser.py`

plotData: run `python plotData.py 0` to display atlantic data. Change arg 0 to 1 for pacific data
