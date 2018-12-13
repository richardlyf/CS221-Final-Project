
#Future vision is how far ahead we want our evaluation function to still evaluate
FUTURE_VISION = 3

#Past vision is how many data points we want to use as our evidence for predicting the future
#Past vision should be at least 2.
PAST_VISION = 2

#How much our grid is scaled relative to latitudes and longitudes (how many latitude/longitudes in a grid box)
GRID_SIZE_SCALE = 1

#Range from 0 to this value of latitudes
LATITUDE_RANGE = 150

#Range from 0 to this value of longitudes
LONGITUDE_RANGE = 150

#Number of particles to simulate for evaluating our Bayes net
PARTICLE_AMOUNT = 20

#Radius of the basic Laplacian, in world grid row/col
LAPLACE_RADIUS = 2

#How much to multiply the LAPLACE_RADIUS by to get the stdev for Gaussian function
LAPLACE_STDEV_FACTOR = 1

#How much we add in our Laplacian
LAPLACE_LAMBDA = 0.02

#If Laplacian will be used in the calculations
USE_LAPLACE = True

#If predicted results will be displayed
VISUAL = True

#Loads pretrained MLE probability table if True
PRETRAINED = True

#Saves trained MLE table is True
SAVE_TRAINED = False
