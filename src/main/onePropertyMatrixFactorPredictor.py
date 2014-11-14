import abstractPredictor
import numpy
from astropy.wcs.docstrings import dims

class OnePropertyMatrixFactorPredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # each region has a different vector for each property
        self.property2region2Vector = {}
        self.property2vector = {}
        # keep this as a backup:
        self.property2median = {}
        
    def predict(self, property, region):
        # it can be the case that we haven't got anything for a coutnry
        if property in self.property2vector and region in self.property2region2Vector[property]:
            return numpy.dot(self.property2vector[property], self.property2region2Vector[property][region])
        else:
            print "no vector for property ", property, " or no vector for region ", region, " for this property"
            return self.property2median[property]
    
    # parameters are: dimensions of vectors, learning rate, reg_parameter, iterations
    def train(self, trainMatrix, textMatrix, params=[10, 0.1, 1, 5000]):
        
        dims, learningRate, regParam, iterations = params
        
        # let's get the median for each property:
        for property, trainRegion2value in trainMatrix.items():
            self.property2median[property] = numpy.median(trainRegion2value.values())
            
        # now let's do the MF for each property separately:
        for property in trainMatrix.keys(): #["/location/statistical_region/population"]:
            print property
            trainRegion2value = trainMatrix[property]
            # first let's filter with MASE
            # anything that is worse than the median predictor (MASE) > 1 should go.
            filteredPatterns = []
            for pattern, region2value in textMatrix.items():
                # make sure that it has at least two value in common with training data, otherwise we might get spurious stuff
                keysInCommon = list(set(region2value.keys()) & set(trainRegion2value.keys()))
                if len(keysInCommon) > 1:
                    #print pattern
                    #print region2value
                    mase = abstractPredictor.AbstractPredictor.MASE(region2value, trainRegion2value)
                    if mase < 0.01:
                        filteredPatterns.append(pattern)
                    
            print "Patterns left after filtering ", len(filteredPatterns)
            if len(filteredPatterns) == 0:
                print "no patterns left after filtering, SKIP"
                continue
            print filteredPatterns
        
            # initialize the low dim representations
            # first the property
            self.property2vector[property] = numpy.random.rand(dims)

            # then the patterns and the regions
            self.property2region2Vector[property] = {}            
            pattern2vector = {}
            valuesPresent = 0
            for pattern in filteredPatterns:
                pattern2vector[pattern] = numpy.random.rand(dims)
                valuesPresent += len(textMatrix[pattern]) 
                for region in textMatrix[pattern].keys():
                    if region not in self.property2region2Vector[property]:
                        self.property2region2Vector[property][region] =  numpy.random.rand(dims)                    
            
            print "Regions after filtering: ", len(self.property2region2Vector[property])
            
            print "values present ", valuesPresent, " density ", float(valuesPresent)/(len(filteredPatterns)*len(self.property2region2Vector[property]))
            
            
            
            # let's go!
            for iter in xrange(iterations):
                # for each property or pattern
                for pp in [property] + filteredPatterns:
                    # we might be getting the values from either the train matrix or the 
                    if pp == property:
                        region2value = trainRegion2value
                    else:
                        region2value = textMatrix[pp]
                    # let's try to reconstruct each known value    
                    for region, value in region2value.items():
                        # we might not have a vector for this region, so ignore
                        if region in self.property2region2Vector[property]:
                            # reconstruction error
                            if pp == property:
                                ppVector = self.property2vector[pp]
                            else:
                                ppVector = pattern2vector[pp]

                            eij = value - numpy.dot(ppVector,self.property2region2Vector[property][region])
                            for k in xrange(dims):
                                ppVector[k] += learningRate * (2 * eij * self.property2region2Vector[property][region][k] - regParam * ppVector[k])
                                self.property2region2Vector[property][region][k] += learningRate * (2 * eij * ppVector[k] - regParam * self.property2region2Vector[property][region][k])        
                    
            
                # let's calculate the squared reconstruction error
                # maybe look only at the training data?
                squaredErrors = []
                preds = {}
                for region, value in trainRegion2value.items():
                    if region in self.property2region2Vector[property]:
                        pred = self.predict(property, region)
                        squaredErrors.append(numpy.square(pred - value))
                        preds[region] = pred
                 
                print "Iteration ", iter, " reconstruction mean squared error on trainMatrix=", numpy.mean(squaredErrors)
                print "Iteration ", iter, " MASE on trainMatrix=", abstractPredictor.AbstractPredictor.MASE(preds, trainRegion2value)
                 
if __name__ == "__main__":
    
    import sys
    # helps detect errors
    numpy.seterr(all='raise')
    # set the random seed for reproducibility
    numpy.random.seed(13)
    
    predictor = OnePropertyMatrixFactorPredictor()
    
    trainMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[1])
    textMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[2])
    testMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[3])

    bestParams = OnePropertyMatrixFactorPredictor.crossValidate(trainMatrix, textMatrix, 4, [[10, 0.000000000001, 0.1, 100000]])
    #predictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)