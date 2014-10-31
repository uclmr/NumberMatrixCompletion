import abstractPredictor
import numpy
import scipy
from copy import copy
import operator
from collections import Counter

# We follow Loh and Wainwright 2012 in constructing the estimators for the missing values 
# It formalizes the intuitions of  Bell & Koren 2007 to construct a and b

class NeighborhoodRegressionPredictor(abstractPredictor.AbstractPredictor):
    
    class PropertyDetails(object):
        
        def __init__(self):
            # these are teh weights
            self.weights = None
            # this keeps the features for each region and property (different properties have different neighbors)
            self.textMatrix = {}
            self.pattern2column = {}
            self.column2pattern = []
            # so we might not have text features for all the regions
            self.region2row = {}
            # keep this as a back up, as well as for scaling purposes.
            self.median = None
    
    
    def __init__(self):
        
        self.property2details = {}

        
    def predict(self, property, region):
        # get the data for the property
        propertyDetails = self.property2details[property]
        # we first need to get the text features for that region and property
        if region in propertyDetails.region2row:# and property =="/location/statistical_region/population":      
            textFeatures = propertyDetails.textMatrix[propertyDetails.region2row[region]]
            # the outcome is just the dot product
            return numpy.dot(textFeatures, propertyDetails.weights)
        else:
            print "No text patterns for region ", region.encode('utf-8'), " returning the median for the property"
            return propertyDetails.median           
    
    def train(self, trainMatrix, textMatrix, params):
        sparse = params[0]
        # set the scaling support parameter
        scalingParam = float(params[1])
        # set the MASE threshold which defines the neighborhood
        maseNeighborhoodThreshold = float(params[2])
        
        minOccurrences = 10

        
        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property, " with params ", params
            propertyDetails = NeighborhoodRegressionPredictor.PropertyDetails()
            self.property2details[property] = propertyDetails
            
            # let's get the median for the property
            propertyDetails.median = numpy.median(trainRegion2value.values())
            print "median ", propertyDetails.median

            # Let's find the neighbors that are good for this property
            print "Original text features: ", len(textMatrix)

            # we also need to keep track of the countsfor each feature in the TRAINING DATA
            patternKept2Counts = {}
            filteredTextMatrix = {}
            #pattern2trainingMedian = {}
            for pattern, region2value in textMatrix.items():
                #print pattern, region2value
                # if we have at least two values in common
                regionsInCommon = set(region2value.keys()) & set(trainRegion2value.keys())
                if len(regionsInCommon) >= minOccurrences:
                    # get the training median for this feature
                    #patternTrainValues = []
                    #for region in regionsInCommon:
                    #    patternTrainValues.append(region2value[region])
                    #pattern2trainingMedian[pattern] = numpy.median(patternTrainValues)
                    
                    scaledMASE = abstractPredictor.AbstractPredictor.supportScaledMASE(region2value, trainRegion2value, scalingParam)
                    if scaledMASE  <= maseNeighborhoodThreshold:
                         filteredTextMatrix[pattern] = copy(region2value)
                         patternKept2Counts[pattern] = len(regionsInCommon)
            print "Text features after removing those with scaled MASE less than ", maseNeighborhoodThreshold, " and at least ", minOccurrences, " occurrences:", len(filteredTextMatrix)
            #print filteredTextMatrix.keys()

            # construct the feature vectors for all the regions
            # these are to be used in prediction time, where the ambiguity of zeros is good.
            # initialize everything to 0
            propertyDetails.textMatrix = []
            for region2value in filteredTextMatrix.values(): 
                for region in region2value:
                    if region not in propertyDetails.region2row:
                        propertyDetails.region2row[region] = len(propertyDetails.textMatrix)
                        propertyDetails.textMatrix.append(scipy.zeros(len(filteredTextMatrix)))

            # add the known values      
            # also estimate the missing proportion
            missingProportions = scipy.zeros(len(filteredTextMatrix))
            for patternNo, (pattern, region2value) in enumerate(filteredTextMatrix.items()):
                propertyDetails.pattern2column[pattern] = patternNo
                propertyDetails.column2pattern.append(pattern)          
                for region, value in region2value.items():
                    propertyDetails.textMatrix[propertyDetails.region2row[region]][patternNo] = value# - pattern2trainingMedian[pattern]
            
                missingProportions[patternNo] = 1 - patternKept2Counts[pattern]/float(len(trainRegion2value))
            #print "missing proportions ",  missingProportions
            #print numpy.shape(missingProportions)
            # Let's construct the linear system
            # A contains the correlations between the features.
            A = scipy.zeros((len(filteredTextMatrix),len(filteredTextMatrix)))
            # b are the transformed values we are mapping to
            b = scipy.zeros(len(filteredTextMatrix))
            
            # So this our y vector
            targetValues = []
            # this is our X*U vector (or Z)
            trainingVectors = []
            for region, value in trainRegion2value.items():
                if region in propertyDetails.region2row:# and property =="/location/statistical_region/population":
                    targetValues.append(value)# - propertyDetails.median)
                    trainingVectors.append(propertyDetails.textMatrix[propertyDetails.region2row[region]])
                else:
                    print "No text patterns for region ", region.encode('utf-8'), " skipping it in training"
            
            # this is the standard solution
            Gamma = numpy.dot(numpy.transpose(trainingVectors), trainingVectors) / float(len(targetValues))
            if sparse:
                # this is the mask matrix
                # off diagonal elements are 
                M  = numpy.outer(1-missingProportions, 1-missingProportions)

                # These should be un-squared
                for i in xrange(len(missingProportions)):
                    M[i][i] = numpy.sqrt(M[i][i])
            
                # following equation 2.11 from Loh and Wainwright    
                A = numpy.divide(Gamma, M)
            else:
                A = Gamma
            
            # again the standard terms
            gamma = numpy.dot(numpy.transpose(trainingVectors), targetValues) / float(len(targetValues))
            
            # following equation 2.11 from Loh and Wainwright
            if sparse:
                b = numpy.divide(gamma, 1-missingProportions)
            else:
                b = gamma
             
            # solve for them using least squares (the exact solver is likely to fail, we have many more features usually)
            # This is not necessarily the best option, as the objective is now non-convex?
            weights = numpy.linalg.lstsq(A, b)[0]
            # TODO: check the condition by calculating the residuals?
            propertyDetails.weights = weights  
                      
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
    #bestParams = neighborhoodRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4, [[False, 1, float("inf")], [True, 1, float("inf")]])
    neighborhoodRegressionPredictor.runEval(trainMatrix, textMatrix, testMatrix, [True, 1, float("inf")])