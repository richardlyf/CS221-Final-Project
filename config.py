
#Future vision is how far ahead we want our evaluation function to still evaluate
FUTURE_VISION = 3

#Past vision is how many data points we want to use as our evidence for predicting the future
#Past vision should be at least 2.
PAST_VISION = 2

#How much our grid is scaled relative to latitudes and longitudes (10 means 50.5W becomes index 505 in our grid)
GRID_SIZE_SCALE = 5

#Range from 0 to this value of latitudes
LATITUDE_RANGE = 150

#Range from 0 to this value of longitudes
LONGITUDE_RANGE = 150

#Number of particles to simulate for evaluating our Bayes net
PARTICLE_AMOUNT = 20

#Radius of the basic Laplacian, in world grid row/col
LAPLACE_RADIUS = 2

#How much we add in our Laplacian
LAPLACE_LAMBDA = 0.02
