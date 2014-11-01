import abstractPredictor
import numpy
import scipy
from copy import copy
import operator
from collections import Counter

class RegressionWithMASE(abstractPredictor.AbstractPredictor):    
    
    def __init__(self):
        
        self.property2median = {}
        # each property has different weights for the features.
        self.property2feature2weight = {}
              
        self.region2feature2value = {}
        
    def predict(self, property, region):

        if region in self.region2feature2value and property =="/location/statistical_region/population":
            # add the median just in case
            prediction = 0.0
            weights = 0.0
            for feature, value in self.region2feature2value[region].items():
                if feature in self.property2feature2weight[property]:
                    # but subtract it from all the features
                    weights += self.property2feature2weight[property][feature]
                    prediction += (value - self.property2median[property])* self.property2feature2weight[property][feature]
            if weights > 0: 
                return prediction/weights + self.property2median[property]
            else:
                print "No active features for region ", region.encode('utf-8'), " returning the median for the property"
                return self.property2median[property]
        else:
            print "No text patterns for region ", region.encode('utf-8'), " returning the median for the property"
            return self.property2median[property]           
    
    def train(self, trainMatrix, textMatrix, params):
        # set the scaling support parameter to calculate MASE
        scalingParam = float(params[0])
        # scaling MASE to a similarity
        simParam = float(params[1])
        # less than this won't make sense 
        minOccurrences = 2

        # add the known values            
        for pattern, region2value in textMatrix.items():
            if len(region2value) >= minOccurrences:
                for region, value in region2value.items():
                    if region not in self.region2feature2value:
                        self.region2feature2value[region] = {}
                self.region2feature2value[region][pattern] = value         

        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property, " with params ", params

            # keep these around to know how put the weights in and out
            pattern2column = {}
            column2pattern = []    

            # let's get the median for the property
            self.property2median[property] = numpy.median(trainRegion2value.values())
            print "median ", self.property2median[property]
            if property == "/location/statistical_region/population":
                # let's filter out features we don't have a lot in the training data.
                filteredTextMatrix = {}
                for patternNo, (pattern, region2value) in enumerate(textMatrix.items()):
                    #print pattern, region2value
                    # if we have at least two values in common
                    regionsInCommon = set(region2value.keys()) & set(trainRegion2value.keys())
                    if len(regionsInCommon) >= minOccurrences and (min(region2value.values()) < max(region2value.values())):# and "population" in pattern:
                        filteredTextMatrix[pattern] = copy(region2value)
                        pattern2column[pattern] = patternNo
                        column2pattern.append(pattern)          
                
                print "Text features after removing those with at least ", minOccurrences, " occurrences:", len(filteredTextMatrix)
                #print column2pattern
                #print filteredTextMatrix.keys()
    
                # So we will construct a system Aw = b, to solve argmin_x(sum(y-wx)^2)
                # A is just XX^T or intuitively the covariance matrix among the features,
                # b is Xy 
                
                # Given that A[i][j] = exp(param*symMAPE(a,1,2,), the diagonal is 1 since symmetric MAPE is 0
                A = scipy.identity(len(filteredTextMatrix))
                
                for i in range(len(filteredTextMatrix)): 
                    for j in range(i+1, len(filteredTextMatrix)):
                        expSymMASE = 0
                        if len(set(filteredTextMatrix[column2pattern[i]].keys()) & set(filteredTextMatrix[column2pattern[j]].keys())) > minOccurrences:
                            #print filteredTextMatrix[column2pattern[i]]
                            #print filteredTextMatrix[column2pattern[j]]
                            MASE1  = abstractPredictor.AbstractPredictor.supportScaledMASE(filteredTextMatrix[column2pattern[i]], filteredTextMatrix[column2pattern[j]], scalingParam)
                            MASE2  = abstractPredictor.AbstractPredictor.supportScaledMASE(filteredTextMatrix[column2pattern[j]], filteredTextMatrix[column2pattern[i]], scalingParam)
                            expSymMASE = numpy.exp(-simParam*(MASE1 + MASE2))
                        A[i][j] = expSymMASE
                        A[j][i] = expSymMASE 
                
                b = scipy.zeros(len(filteredTextMatrix))
                
                for i in range(len(filteredTextMatrix)):
                    MASE1  = abstractPredictor.AbstractPredictor.supportScaledMASE(filteredTextMatrix[column2pattern[i]], trainRegion2value, scalingParam)
                    MASE2  = abstractPredictor.AbstractPredictor.supportScaledMASE(trainRegion2value, filteredTextMatrix[column2pattern[i]], scalingParam)
                    b[i] = numpy.exp(-simParam*(MASE1 + MASE2))
                print "constructed the system of equations"
                             
                # solve for them using least squares (the exact solver is likely to fail, we have many more features usually)
                # This is not necessarily the best option, as the objective is now non-convex?
                weights, res = scipy.optimize.nnls(A, b)
                print weights
                print res
                self.property2feature2weight[property] = {}
                for i, weight in enumerate(weights):
                    #print i, weight, column2pattern[i]                    
                    self.property2feature2weight[property][column2pattern[i]] = weight
                    
                sortedPatterns = sorted(self.property2feature2weight[property].items(), key=operator.itemgetter(1), reverse=True)
                print "top-100 patterns found"
                print sortedPatterns[100]
                      
if __name__ == "__main__":
    
    import sys
    # underflows are usually small values that go to 0. These usually do not matter unless they are dividers in which case an error is raised.
    numpy.seterr(all='raise')
    numpy.seterr(under='ignore')
    regressionWithMASE = RegressionWithMASE()
    
    trainMatrix = regressionWithMASE.loadMatrix(sys.argv[1])
    textMatrix = regressionWithMASE.loadMatrix(sys.argv[2])
    testMatrix = regressionWithMASE.loadMatrix(sys.argv[3])
    
    # if the first param is False, the other are ignored and no neighborhood is active.
    # thus it is the same as regression 0, True
    #bestParams = neighborhoodRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4, [[False, 1, float("inf")], [True, 1, float("inf")]])
    regressionWithMASE.runEval(trainMatrix, textMatrix, testMatrix, [1, 1])