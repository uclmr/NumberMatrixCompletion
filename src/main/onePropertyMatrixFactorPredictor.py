import abstractPredictor
import numpy
import multiprocessing
import copy
import operator
from __builtin__ import False
    

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
            print "no vector for property ", property.encode('utf-8'), " or no vector for region ", region.encode('utf-8'), " for this property"
            return self.property2median[property]
        
    def trainRelation(self, d, property, trainMatrix, textMatrix, learningRate, regParam, iterations):
        #property = propertyQueue.get()
        trainRegion2value = trainMatrix[property]
        print property, " training starting now"
        median = numpy.median(trainRegion2value.values())
        medianErrors = []
        for value in trainRegion2value.values():
            medianErrors.append(numpy.abs(median - value))
        medianError = numpy.mean(medianErrors)
        
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
                if mase < 0.1:
                    filteredPatterns.append(pattern)
                
        print property, ", patterns left after filtering ", len(filteredPatterns)
        if len(filteredPatterns) == 0:
            print property, ", no patterns left after filtering, SKIP"
            d[property] = (median, None, None)
            return
        print filteredPatterns
        
        # ignore the setting, set it according to the text patterns   
        dims = max(2, int(numpy.ceil(numpy.sqrt(len(filteredPatterns)))))
        print property, ", set the dimensions to the square root of the text patterns = ", dims 
    
        #regParam /= numpy.power(dims, 0.1)
        #print property, "set the reg param to ", regParam
    
        # initialize the low dim representations
        # first the property
        propertyVector = numpy.random.rand(dims)

        # then the patterns and the regions
        region2Vector = {}            
        pattern2vector = {}
        valuesPresent = 0
        for pattern in filteredPatterns:
            pattern2vector[pattern] = numpy.random.rand(dims)
            valuesPresent += len(textMatrix[pattern]) 
            for region in textMatrix[pattern].keys():
                if region not in region2Vector:
                    region2Vector[region] =  numpy.random.rand(dims)                    
        
        print property, ", regions after filtering: ", len(region2Vector)
        
        print property, ", values present ", valuesPresent, " density ", float(valuesPresent)/(len(filteredPatterns)*len(region2Vector))
        
        # let's go!
        for iter in xrange(iterations):
            # for each property or pattern
            allpps = [property] + filteredPatterns
            numpy.random.shuffle(allpps)            
            for pp in allpps:
                # we might be getting the values from either the train matrix or the 
                if pp == property:
                    region2value = trainRegion2value
                else:
                    region2value = textMatrix[pp]
                # let's try to reconstruct each known value    
                regVals = region2value.items()
                numpy.random.shuffle(regVals)
                for region, value in regVals:
                    # we might not have a vector for this region, so ignore
                    if region in region2Vector:
                        # reconstruction error
                        if pp == property:
                            ppVector = propertyVector
                        else:
                            ppVector = pattern2vector[pp]

                        eij = value - numpy.dot(ppVector,region2Vector[region])
                        # this is equivalent to dividing all entries in the matrix by this factor. Would be more efficient to do it like this I guess
                        #eij /= medianError
                        # this is essentially L1 loss
                        eij = numpy.sign(eij)
                        #if numpy.abs(eij) < 1:
                        ppVector += learningRate * (2 * eij * region2Vector[region] - regParam * ppVector)
                        region2Vector[region] += learningRate * (2 * eij * ppVector - regParam * region2Vector[region])
                        #else:
                        #    ppVector += learningRate * (2 * numpy.sign(eij) * region2Vector[region] - regParam * ppVector)
                        #    region2Vector[region] += learningRate * (2 * numpy.sign(eij) * ppVector - regParam * region2Vector[region])                            
        
            # let's calculate the squared reconstruction error
            # maybe look only at the training data?
            squaredErrors = []
            absoluteErrors = []
            preds = {}
            for region, value in trainRegion2value.items():
                if region in region2Vector:
                    pred = numpy.dot(propertyVector,region2Vector[region])
                    try:
                        error = pred - value
                        absoluteErrors.append(numpy.absolute(error))
                        squaredErrors.append(numpy.square(error))
                    except FloatingPointError:
                        print property, ", iteration ", iter, ", error for region ", region.encode('utf-8'), " too big, IGNORED"
                    preds[region] = pred
            mase = abstractPredictor.AbstractPredictor.MASE(preds, trainRegion2value)
            print property, ", iteration ", iter, " reconstruction mean squared error on trainMatrix=", numpy.mean(squaredErrors)
            print property, ", iteration ", iter, " reconstruction mean absolute error on trainMatrix=", numpy.mean(absoluteErrors)
            print property, ", iteration ", iter, " MASE on trainMatrix=", mase
            
            euclidDistanceFromPropertyVector = {}
            pVectorSquare = numpy.dot(propertyVector, propertyVector)
            for pattern, vector in pattern2vector.items():
                # if the distance is too high ignore.
                try:
                    euclidDistanceFromPropertyVector[pattern] = numpy.sqrt(numpy.dot(vector, vector) - 2 * numpy.dot(vector, propertyVector) + pVectorSquare)
                except FloatingPointError:
                    pass
            
            sortedPaterns= sorted(euclidDistanceFromPropertyVector.items(), key=operator.itemgetter(1))
            
            print "top-10 patterns closest to the property in euclidean distance"
            for idx in xrange(min(10, len(sortedPaterns))):
                print sortedPaterns[idx][0].encode('utf-8'), ":", sortedPaterns[idx][1]                 
            
            if mase < 0.000001:
                break
        
        d[property] = (median, propertyVector, region2Vector)
        #self.property2vector[property] = propertyVector
        #self.property2region2Vector[property] = region2Vector 
        
                    
    
    # parameters are: dimensions of vectors, learning rate, reg_parameter, iterations
    def train(self, trainMatrix, textMatrix, params=[10, 0.1, 1, 5000]):
    
        dims, learningRate, regParam, iterations = params                    

        #propertyQueue = multiprocessing.Queue(maxsize=0)
        #num_threads = 3

        #for i in range(num_threads):
        #    worker = multiprocessing.Process(target=self.trainRelation, args=(propertyQueue, trainMatrix, textMatrix, learningRate, regParam, iterations,))
            #worker.daemon = True
        #    worker.start()

        # we need q queue for the the results to be put
        
        mgr = multiprocessing.Manager()
        d = mgr.dict()
         

        # now let's do the MF for each property separately:
        jobs = []
        for property in trainMatrix.keys(): #, "/location/statistical_region/renewable_freshwater_per_capita"]: #  
            #if property in ["/location/statistical_region/fertility_rate", "/location/statistical_region/population"]:
            job = multiprocessing.Process(target=self.trainRelation, args=(d, property, trainMatrix, textMatrix, learningRate, regParam, iterations,))
            jobs.append(job)
            #else:
            #    self.property2median[property] = numpy.median(trainMatrix[property].values())
        
        # Start the processes (i.e. calculate the random number lists)        
        for j in jobs:
            j.start()

        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
            
        for property, (median, propertyVector, region2Vector) in d.items():
            self.property2median[property] = copy.copy(median)
            if region2Vector != None:    
                self.property2region2Vector[property] = copy.copy(region2Vector)
                self.property2vector[property] = copy.copy(propertyVector)
            
        
        print self.property2median
        
        print "Done training"
        
                 
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

    bestParams = OnePropertyMatrixFactorPredictor.crossValidate(trainMatrix, textMatrix, 4, [[100, 0.0001, 0.1, 10000]])
    #predictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)