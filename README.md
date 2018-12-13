# CS221-Final-Project


## Files:

#### Programs:

baseline.py: runs the baseline linear extrapolator

config.py: configurations and hyperparameters

dataParser.py: handles reading and plotting CSV file contents.

mle.py: body of the MLE and the MLE+Laplace

partitionData.py: splits dataset into Train, Validation, and Test.

plotData.py: displays the entire atlantic dataset on the map

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


MLE: run `python3 mle.py`

