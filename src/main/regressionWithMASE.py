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

        if region in self.region2feature2value:# and property =="/location/statistical_region/population":
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
    
    # returns the average of the MASE both ways between two dicts.
    # should get the same result (MASE(dict1, dict2) + MASE(dict2, dict1))/2, but faster
    @staticmethod
    def symmetrizedMASE(dict1, dict2):
        # get the scaling factor for MASE1
        # first let's estimate the error from the median:
        median1 = numpy.median(dict1.values())
                                    
        # calculate the errors of the test median
        medianAbsErrors1 = []
        for value in dict1.values():
            medianAbsErrors1.append(abs(value - median1))
        meanMedianError1 = numpy.mean(medianAbsErrors1)

        # get the scaling factor for MASE2
        median2 = numpy.median(dict2.values())
        # calculate the errors of the test median
        medianAbsErrors2 = []
        for value in dict2.values():
            medianAbsErrors2.append(abs(value - median2))
        
        # get the scaling factor for MASE2
        meanMedianError2 = numpy.mean(medianAbsErrors2)

        # get the regions in common these features have
        regionsInCommon = set(dict1.keys()) & set(dict2.keys())

        # compute the  errors
        predAbsErrors = []
        for key in regionsInCommon:
            predAbsErrors.append(abs(dict2[key] - dict1[key]))
                                    
        # unscaled MASEs        
        MASE1 = numpy.mean(predAbsErrors)/meanMedianError1
        MASE2 = numpy.mean(predAbsErrors)/meanMedianError2

        return (MASE1+MASE2)/2.0
        
    
    def train(self, trainMatrix, textMatrix, params):
        # set the scaling support parameter to calculate MASE
        scalingParam = float(params[0])
        # scaling MASE to a similarity
        simParam = float(params[1])
        # less than this won't make sense to calculate MASE 
        minOccurrences = 2

        # change the text matrix from pattern2region2value to region2feature2value
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
            #if property == "/location/statistical_region/population":
            # let's filter out features we don't have a lot in the training data.
            filteredTextMatrix = {}
            for patternNo, (pattern, region2value) in enumerate(textMatrix.items()):
                #print pattern, region2value
                # if we have at least two values in common with the training data
                regionsInCommon = set(region2value.keys()) & set(trainRegion2value.keys())
                # also, avoid features that the same value everywhere.
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
                        
                    feature1 = filteredTextMatrix[column2pattern[i]]
                    feature2 = filteredTextMatrix[column2pattern[j]]
                    # get the regions in common these features have
                    regionsInCommon = set(feature1.keys()) & set(feature2.keys())
                    # if they have enough in common
                    if len(regionsInCommon) > minOccurrences:
                        symMASE = self.symmetrizedMASE(feature1,feature2)
                        scalingFactor = float(scalingParam)/(scalingParam + len(regionsInCommon))
        
                        expSymMASE = numpy.exp(-simParam*scalingFactor*symMASE)
                    A[i][j] = expSymMASE
                    A[j][i] = expSymMASE 
                
            b = scipy.zeros(len(filteredTextMatrix))
                
            for i in range(len(filteredTextMatrix)):
                # a feature that doesn'have enoguh in common with the property we are trying to predict
                # must have been filtered out already
                regionsInCommon = set(filteredTextMatrix[column2pattern[i]].keys()) & set(trainRegion2value.keys())
                scalingFactor = float(scalingParam)/(scalingParam + len(regionsInCommon))
                symMASE = self.symmetrizedMASE(filteredTextMatrix[column2pattern[i]], trainRegion2value)
                b[i] = numpy.exp(-simParam*scalingFactor*symMASE)
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
            print sortedPatterns[:100]
                      
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
    bestParams = regressionWithMASE.crossValidate(trainMatrix, textMatrix, 4, [[1,1], [2,1], [0.5,1]])
    regressionWithMASE.runEval(trainMatrix, textMatrix, testMatrix, bestParams)