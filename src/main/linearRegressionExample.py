import numpy
import scipy
from sklearn.linear_model import *

# random seed for reproducibility
numpy.random.seed(13)

# just to have more readable output
numpy.set_printoptions(suppress=True)

print "Creating the 10 target values (instances)"
print "generated from uniform distribution [-50,50)"
targets = (numpy.random.rand(10) -0.5)* 100
print targets

print "Creating the features"
print "5 perfect predictors, and 5 random (irrelevant) ones"
features = []
for target in targets:
    features.append(numpy.concatenate((numpy.ones(5)*target, numpy.random.rand(5))))
features = numpy.multiply(numpy.random.rand(10,10) >= 0.0, features)
print str(features)

# standard linear regressor
print "Training a standard linear regressor"
stdRegressor = LinearRegression()
stdRegressor.fit(features, targets)
print "weights learned"
print stdRegressor.coef_
print "the weights (coefficients) are OK more or less, the 5 good features get 1/5 weight, the irrelevant ones are 0"

# l1 (sparse) regressor
print "Training an l1 (sparse) linear regressor"
l1Regressor = Lasso() 
l1Regressor.fit(features, targets)
print "weights learned"
print l1Regressor.coef_
print "the weights for the irrelevant features are 0"
print "also, since it prefers sparse solutions, it picks only one of the good features to give almost 1, the rest are 0"
print "this is troublesome when data is missing during testing"

# missing half the features
print "Zeroing half the features"
print "since the data has 0 mean, implicilty we also impute the data with the mean"
missingHalfFeatures = numpy.multiply(numpy.random.rand(10,10) > 0.5, features)
print missingHalfFeatures
print "Training standard linear regressor"
stdRegressor.fit(missingHalfFeatures, targets)
print "weights learned"
print stdRegressor.coef_

# missing 90% the features
print "Zeroing 90% the features"
missing90pcFeatures = numpy.multiply(numpy.random.rand(10,10) > 0.9, features)
print missing90pcFeatures
stdRegressor.fit(missing90pcFeatures, targets)
print "weights learned"
print stdRegressor.coef_

print "as the data goes missing more and more, the weights become rapidly worse"
