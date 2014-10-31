import abstractPredictor
import numpy
import scipy
from sklearn.linear_model import *
from copy import copy
import operator

class SGDRegressionPredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # this keeps the regression model
        self.property2regressor = {}
        # this keeps the features for each region
        # this might include some kind of imputation. eg. with the mean? 
        self.textMatrix = None
        self.pattern2column = {}
        self.column2pattern = []
        self.region2row = {}
        # we use the median as back-off in case there are not text patterns for that region
        self.property2median = {}
        
    def predict(self, property, region):
        # we first need to get the text features for that region
        if region in self.region2row:
            textFeatures = self.textMatrix[self.region2row[region]]
            # if we are fixing the bias to the median
            return self.property2regressor[property].predict(textFeatures)
        else:
            print "No text patterns for region ", region.encode('utf-8'), " returning the median for the property"
            return self.property2median[property]           
    
    def train(self, trainMatrix, textMatrix, params):

        epochs = params[0]
        initialLearningRage = params[1]
        
        self.textMatrix = []
                
        for pattern, region2value in textMatrix.items():
            filteredTextMatrix[pattern] = copy(region2value)
        #textMatrix = filteredTextMatrix
        
        # initialize to 0 all the feature vectors
        # This should have all the regions we have collected data for
        for region2value in filteredTextMatrix.values():
            for region in region2value:
                if region not in self.region2row:
                    self.region2row[region] = len(self.textMatrix)
                    self.textMatrix.append(scipy.zeros(len(textMatrix)))
        
        # add the known values, missing ones will remain 0
        for patternNo, (pattern, region2value) in enumerate(filteredTextMatrix.items()):
            self.pattern2column[pattern] = patternNo
            self.column2pattern.append(pattern)          
            for region, value in region2value.items():
                # if bias is fixed, subtract the median
                self.textMatrix[self.region2row[region]][patternNo] = value
        
        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property, " with params " , params
            # using the median of the property as back up
            self.property2median[property] = numpy.median(trainRegion2value.values())
            
            self.property2regressor[property] = SGDRegressor(loss='huber', eta0=initialLearningRage, fit_intercept=False, verbose=1, n_iter=epochs, penalty='elasticnet')
             
            # first construct the target values
            targetValues = []
            # occasionally we have missing values
            trainingVectors = []
            for region, value in trainRegion2value.items():
                if region in self.region2row:
                    targetValues.append(value)
                    trainingVectors.append(self.textMatrix[self.region2row[region]])
                else:
                    print "No text patterns for region ", region.encode('utf-8'), " skipping it in training"
            # Code to inspect the results
            #print trainingVectors
            #print targetValues
            #print numpy.any(numpy.isnan(trainingVectors)) #False
            #print numpy.any(numpy.isinf(trainingVectors)) #False
            #print numpy.all(numpy.isfinite(trainingVectors)) #True
            self.property2regressor[property].fit(trainingVectors, targetValues)
                # this should print the weights learnt
            #    pattern2weight = {}
            #    for idx, pattern in enumerate(self.column2pattern):
            #        pattern2weight[pattern] = self.property2regressor[property].coef_[idx]
            #    sortedWeights = sorted(pattern2weight.items(), key=operator.itemgetter(1))
            #    for weight in sortedWeights:
            #        print weight
            #    print "intercept:", self.property2regressor[property].intercept_
        
          
if __name__ == "__main__":
    
    import sys
    # underflows are usually small values that go to 0. These usually do not matter unless they are dividers in which case an error is raised.
    numpy.seterr(all='raise')
    numpy.seterr(under='ignore')
    sgdRegressionPredictor = SGDRegressionPredictor()
    
    trainMatrix = sgdRegressionPredictor.loadMatrix(sys.argv[1])
    textMatrix = sgdRegressionPredictor.loadMatrix(sys.argv[2])
    testMatrix = sgdRegressionPredictor.loadMatrix(sys.argv[3])
    
    # So far it seems like fixing the bias to the median and keeping everything works better.
    bestParams = sgdRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4 ,[[1000, 1]])
    sgdRegressionPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)