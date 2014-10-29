import abstractPredictor
import numpy
import scipy
from sklearn.linear_model import *
from copy import copy
import operator

class NeighborhoodRegressionPredictor(abstractPredictor.AbstractPredictor):
    
    class PropertyDetails(object):
        
        def __init__(self):
            # this keeps the regression model
            self.regressor = None
            # this keeps the features for each region and property (different properties have different neighbors)
            self.textMatrix = {}
            self.pattern2column = {}
            self.column2pattern = []
            # so we might not have text features for all the regions
            self.region2row = {}
            self.median = None
    
    
    def __init__(self):
        
        self.property2details = {}

        
    def predict(self, property, region):
        # get the data for the property
        propertyDetails = self.property2details[property]
        # we first need to get the text features for that region and property
        if region in propertyDetails.region2row:# and property =="/location/statistical_region/population":      
            textFeatures = propertyDetails.textMatrix[propertyDetails.region2row[region]]
            return propertyDetails.median + propertyDetails.regressor.predict(textFeatures)
        else:
            print "No text patterns for region ", region.encode('utf-8'), " returning the median for the property"
            return propertyDetails.median           
    
    def train(self, trainMatrix, textMatrix, params):
        # use neighborhood to filter?
        neighborhood = params[0]
        # set the scaling support parameter
        scalingParam = float(params[1])
        # set the MASE threshold which defines the neighborhood
        maseNeighborhoodThreshold = float(params[2])
        
        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property, " with params ", params
            propertyDetails = NeighborhoodRegressionPredictor.PropertyDetails()
            self.property2details[property] = propertyDetails
            
            # let's get the median for the property
            propertyDetails.median = numpy.median(trainRegion2value.values())
            print "median ", propertyDetails.median
            # do not learn the intercept, we will use the bias
            propertyDetails.regressor = LinearRegression(fit_intercept=False)

            # Let's find the neighbors that are good for this property
            print "Original text features: ", len(textMatrix)
            if neighborhood:
                filteredTextMatrix = {}
                for pattern, region2value in textMatrix.items():
                    #print pattern, region2value
                    # if we have at least two values in common
                    if len(set(region2value.keys()) & set(trainRegion2value.keys())) >= 2:
                        scaledMASE = abstractPredictor.AbstractPredictor.supportScaledMASE(region2value, trainRegion2value, scalingParam)
                        if scaledMASE  <= maseNeighborhoodThreshold:
                            filteredTextMatrix[pattern] = copy(region2value)
                print "Text features after removing those with scaled MASE less than ", maseNeighborhoodThreshold, " and at least 2 occurrences:", len(filteredTextMatrix)
                print filteredTextMatrix.keys()
            else:
                filteredTextMatrix = copy(textMatrix)

            # construct the feature vectors for all the regions
            # initialize everything to 0
            propertyDetails.textMatrix = []
            for region2value in filteredTextMatrix.values(): 
                for region in region2value:
                    if region not in propertyDetails.region2row:
                        propertyDetails.region2row[region] = len(propertyDetails.textMatrix)
                        propertyDetails.textMatrix.append(scipy.zeros(len(filteredTextMatrix)))

            # add the known values            
            for patternNo, (pattern, region2value) in enumerate(filteredTextMatrix.items()):
                #remove the median from everything so that the missing values are imputed by zero.
                patternMedian = numpy.median(region2value.values())
                propertyDetails.pattern2column[pattern] = patternNo
                propertyDetails.column2pattern.append(pattern)          
                for region, value in region2value.items():
                    propertyDetails.textMatrix[propertyDetails.region2row[region]][patternNo] = value - patternMedian
            
            # construct the training data
            targetValues = []
            trainingVectors = []
            for region, value in trainRegion2value.items():
                if region in propertyDetails.region2row:# and property =="/location/statistical_region/population":
                    targetValues.append(value - propertyDetails.median)
                    trainingVectors.append(propertyDetails.textMatrix[propertyDetails.region2row[region]])
                else:
                    print "No text patterns for region ", region.encode('utf-8'), " skipping it in training"

            propertyDetails.regressor.fit(trainingVectors, targetValues)
                      
if __name__ == "__main__":
    
    import sys
    # underflows are usually small values that go to 0. These usually do not matter unless they are dividers in which case an error is raised.
    numpy.seterr(all='raise')
    numpy.seterr(under='ignore')
    neighborhoodRegressionPredictor = NeighborhoodRegressionPredictor()
    
    trainMatrix = neighborhoodRegressionPredictor.loadMatrix(sys.argv[1])
    textMatrix = neighborhoodRegressionPredictor.loadMatrix(sys.argv[2])
    testMatrix = neighborhoodRegressionPredictor.loadMatrix(sys.argv[3])
    
    # if the first param is False, the other are ignored and no neighborhood is active.
    # thus it is the same as regression 0, True
    bestParams = neighborhoodRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4 ,[[False, 0, 0],[True, 0.5, 0.1],[True, 0.5, 0.05],[True, 0.5, 0.025]])
    neighborhoodRegressionPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)