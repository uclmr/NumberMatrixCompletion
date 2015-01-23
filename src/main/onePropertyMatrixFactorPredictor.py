import fixedValuePredictor
import numpy
import multiprocessing
import copy
import operator
from collections import Counter, OrderedDict
    

class OnePropertyMatrixFactorPredictor(fixedValuePredictor.FixedValuePredictor):
    
    def __init__(self):
        # each region has a different vector for each property
        self.property2region2Vector = {}
        self.property2vector = {}
        # keep this as a backup:
        fixedValuePredictor.FixedValuePredictor.__init__(self)
        # This is the scaling factor
        self.scalingFactor = None
        
    def predict(self, property, region, of):
        # it can be the case that we haven't got anything for a country
        if property in self.property2vector and region in self.property2region2Vector[property]:
            # remember to mulitply with the scaling factor
            return numpy.dot(self.property2vector[property], self.property2region2Vector[property][region]) * self.scalingFactor
        else:
            of.write("no vector for property " + property.encode('utf-8') + " or no vector for region " + region.encode('utf-8') + " for this property\n")
            return fixedValuePredictor.FixedValuePredictor.predict(self, property, region)
        
    def trainRelation(self, property, trainRegion2value, textMatrix, of, learningRate, regParam, iterations, filterThreshold, learningRateBalance, scale):

        # get the back up fixed values
        fixedValuePredictor.FixedValuePredictor.trainRelation(self, property, trainRegion2value, textMatrix)
                
        # first let's filter 
        filteredPatterns = []
        filteredPatternMAPES = []

        scaling = scale
        # if scaling
        if scaling:
            self.scalingFactor = max(numpy.abs(numpy.min(trainRegion2value.values())), numpy.abs(numpy.max(trainRegion2value.values())))
            of.write("scaling factor:" + str(self.scalingFactor) + "\n")
        else:
            self.scalingFactor = 1.0            
        
        # construct the (possibly scaled) text matrix, but before scaling, just in case the filtering mechanism cares about it
        scaledTextMatrix = {}
        for pattern, region2value in textMatrix.items():
            # make sure that it has at least two value in common with training data, otherwise we might get spurious stuff
            keysInCommon = list(set(region2value.keys()) & set(trainRegion2value.keys()))
            if len(keysInCommon) > 1:
                #print pattern
                #print region2value
                mape = self.supportScaledMAPE(region2value, trainRegion2value, 1)
                if mape < filterThreshold:
                    filteredPatterns.append(pattern)
                    filteredPatternMAPES.append(mape)
                    # scale if necessary
                    if scaling:
                        scaledRegion2value = copy.deepcopy(region2value)
                        for region, value in region2value.items():
                            scaledRegion2value[region] = value/self.scalingFactor
                        scaledTextMatrix[pattern] = scaledRegion2value
                    else:
                        scaledTextMatrix[pattern] = region2value
                    # order it so that we don't have issues with random init           
                    scaledTextMatrix[pattern] = OrderedDict(sorted(scaledTextMatrix[pattern].items(), key=lambda t: t[0]))
        of.write(property+ ", patterns left after filtering " + str(len(filteredPatterns)) + "\n")
        if len(filteredPatterns) == 0:
            of.writeln(property, ", no patterns left after filtering, SKIP")
            return
        
        # if scaling
        if scaling:
            # replace (without over-writing) the original training values
            scaledTrainRegion2value = {}
            for region, value in trainRegion2value.items():
                scaledTrainRegion2value[region] = value/self.scalingFactor            
        else:
            scaledTrainRegion2value = trainRegion2value
        
        # order it so that we don't have issues with random init 
        scaledTrainRegion2value = OrderedDict(sorted(scaledTrainRegion2value.items(), key=lambda t: t[0]))
        
        
        of.write(property + " training starting now\n")
        
        # set it according to the text patterns   
        dims = max(2, int(numpy.ceil(numpy.sqrt(len(filteredPatterns)))))
        of.write(property + ", set the dimensions to the square root of the text patterns = " + str(dims) + "\n") 
    
        #regParam /= numpy.power(dims, 0.1)
        #print property, "set the reg param to ", regParam

        # set the random seed for reproducibility
        numpy.random.seed(13)
    
        # initialize the low dim representations
        # first the property
        propertyVector = numpy.random.rand(dims)
        

        # then the patterns and the regions
        region2Vector = {}            
        pattern2vector = {}
        # also count the times they appear in the patterns
        trainingRegion2counts = Counter()
        valuesPresent = 0
        for pattern in filteredPatterns:
            pattern2vector[pattern] = numpy.random.rand(dims)
            valuesPresent += len(scaledTextMatrix[pattern]) 
            for region in scaledTextMatrix[pattern].keys():
                if region not in region2Vector:
                    region2Vector[region] =  numpy.random.rand(dims)
                if region in scaledTrainRegion2value:
                    trainingRegion2counts[region] += 1 
                    
        
        of.write(property + ", regions after filtering: " + str(len(region2Vector)) + "\n")
        
        of.write(property + ", values present " + str(valuesPresent) + " density " + str(float(valuesPresent)/(len(filteredPatterns)*len(region2Vector))) + "\n")
                
        allpps = [property] + filteredPatterns

        # calculate the initial trainError just to make sure we have the same starting point
        absoluteErrors = []
        for region, value in trainRegion2value.items():
            if region in region2Vector:
                # remember to multiply with the scaling factor
                pred = numpy.dot(propertyVector,region2Vector[region]) * self.scalingFactor
                error = pred - value 
                absoluteErrors.append(numpy.absolute(error))
        of.write(property + ", initial reconstruction mean absolute error on trainMatrix=" + str(numpy.mean(absoluteErrors)) + "\n") 
        
        
        for iter in xrange(iterations):
            numpy.random.shuffle(allpps)            
            for pp in allpps:
                # we might be getting the values from either the train matrix or the 
                if pp == property:
                    region2value = scaledTrainRegion2value
                else:
                    region2value = scaledTextMatrix[pp]
                # let's try to reconstruct each known value    
                regVals = region2value.items()
                numpy.random.shuffle(regVals)
                for region, value in regVals:
                    # we might not have a vector for this region, so ignore
                    if region in region2Vector:
                        # reconstruction error
                        if pp == property:
                            ppVector = propertyVector
                            # +1 is for the cases where we haven't seen this training region with any pattern
                            lr = (learningRateBalance*trainingRegion2counts[region] + 1) * learningRate
                        else:
                            ppVector = pattern2vector[pp]
                            lr = learningRate

                        # so this the possibly scaled reconstruction error
                        eij = value - numpy.dot(ppVector,region2Vector[region])
                                                    
                        ppVector += lr * (2 * eij * region2Vector[region] - regParam * ppVector)
                        region2Vector[region] += lr * (2 * eij * ppVector - regParam * region2Vector[region])
        
            # let's calculate the squared reconstruction error
            # maybe look only at the training data?
            #squaredErrors = []
            absoluteErrors = []
            preds = {}
            for region, value in trainRegion2value.items():
                if region in region2Vector:
                    pred = numpy.dot(propertyVector,region2Vector[region]) * self.scalingFactor
                    try:
                        error = pred - value
                        absoluteErrors.append(numpy.absolute(error))
                        #squaredErrors.append(numpy.square(error))
                    except FloatingPointError:
                        of.write(property + ", iteration " + str(iter) + ", error for region " + region.encode('utf-8') + " too big, IGNORED\nnnnn.,mn,,m.n,,.,./..///////,.m/.,")
                    preds[region] = pred
            #mase = self.MASE(preds, trainRegion2value)
            mape = self.MAPE(preds, trainRegion2value)
            #print property, ", iteration ", iter, " reconstruction mean squared error on trainMatrix=", numpy.mean(squaredErrors)
            of.write(property + ", iteration " + str(iter) + " reconstruction mean absolute error on trainMatrix=" + str(numpy.mean(absoluteErrors)) + "\n")
            #print property, ", iteration ", iter, " MASE on trainMatrix=", mase
            # MAPE ignores scale
            of.write(property + ", iteration " + str(iter) + " MAPE on trainMatrix=" + str(mape) + "\n")

            #patternSquaredErrors = []
            patternAbsoluteErrors = []
            trueVals = {}
            predVals = {}            
            for pattern in filteredPatterns:
                region2value = textMatrix[pattern]
                for region, value in region2value.items():
                    pred = numpy.dot(pattern2vector[pattern],region2Vector[region])  * self.scalingFactor
                    error = pred - value
                    patternAbsoluteErrors.append(numpy.absolute(error))
                    trueVals[region+pattern] = value
                    predVals[region+pattern] = pred
            #print property, ", iteration ", iter, " reconstruction mean squared error on textMatrix=", numpy.mean(patternSquaredErrors)
            textMean = numpy.mean(patternAbsoluteErrors)
            of.write(property + ", iteration " + str(iter) + " reconstruction mean absolute error on textMatrix=" + str(textMean) + "\n") 
            patternMape = self.MAPE(predVals, trueVals)
            of.write(property + ", iteration " + str(iter) + " MAPE on textMatrix=" + str(patternMape) + "\n") 
            
            euclidDistanceFromPropertyVector = {}
            pVectorSquare = numpy.dot(propertyVector, propertyVector)
            for pattern, vector in pattern2vector.items():
                # if the distance is too high ignore.
                try:
                    euclidDistanceFromPropertyVector[pattern] = numpy.sqrt(numpy.dot(vector, vector) - 2 * numpy.dot(vector, propertyVector) + pVectorSquare)
                except FloatingPointError:
                    pass
            
            sortedPaterns= sorted(euclidDistanceFromPropertyVector.items(), key=operator.itemgetter(1))
            
            of.write("top-10 patterns closest to the property in euclidean distance : distance from property\n")
            for idx in xrange(min(10, len(sortedPaterns))):
                of.write(sortedPaterns[idx][0].encode('utf-8') + ":" +  str(sortedPaterns[idx][1])+ "\n")
            
            if mape < 0.000001:
                break
        
        #d[property] = (propertyVector, region2Vector)
        self.property2vector[property] = propertyVector
        self.property2region2Vector[property] = region2Vector 
        
                    
    
    # parameters are: learning rate, reg_parameter, iterations, filtering threshold
    def train(self, trainMatrix, textMatrix, params=[0.1, 1, 5000, 0.1, 1]):
        # get the back up fixed values
        fixedValuePredictor.FixedValuePredictor.train(self, trainMatrix, textMatrix)
    
        learningRate, regParam, iterations, filterThreshold, learningRateBalance = params                    
        
        mgr = multiprocessing.Manager()
        d = mgr.dict()
         

        # now let's do the MF for each property separately:
        jobs = []
        for property in trainMatrix.keys(): # ["/location/statistical_region/renewable_freshwater_per_capita", "/location/statistical_region/population"]: # ["/location/statistical_region/size_of_armed_forces"]:#    
            #if property in ["/location/statistical_region/fertility_rate"]: # 
            job = multiprocessing.Process(target=self.trainRelation, args=(d, property, trainMatrix, textMatrix, learningRate, regParam, iterations, filterThreshold, learningRateBalance,))
            jobs.append(job)
            #else:
            #    self.property2median[property] = numpy.median(trainMatrix[property].values())
        
        # Start the processes (i.e. calculate the random number lists)        
        for j in jobs:
            j.start()

        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
            
        for property, (propertyVector, region2Vector) in d.items():    
            self.property2region2Vector[property] = copy.copy(region2Vector)
            self.property2vector[property] = copy.copy(propertyVector)
        
        print "Done training"
        
                 
if __name__ == "__main__":
    
    import sys
    # helps detect errors
    numpy.seterr(all='raise')
    
    predictor = OnePropertyMatrixFactorPredictor()
    
    trainMatrix = predictor.loadMatrix(sys.argv[1])
    textMatrix = predictor.loadMatrix(sys.argv[2])
    testMatrix = predictor.loadMatrix(sys.argv[3])
    
    outputFileName = sys.argv[4]

    learningRates = [0.0001]
    l2penalties = [0.1,0.2,0.3]
    iterations =  [1000, 2000, 4000, 8000, 10000]
    filterThresholds = [0.02,0.04]
    learningRateBalances = [0.0, 1.0]
    scale = [False, True]
    
    # These are the winning ones:
    #learningRates = [0.0001]
    #l2penalties = [0.3]
    #iterations = [10000]
    #filterThresholds = [0.02]
    #learningRateBalances = [0.0, 1.0]
    
    # this loads all relations
    #properties = json.loads(open("/cs/research/intelsys/home1/avlachos/FactChecking/featuresKept.json"))
    # Othewise, specify which ones are needed:
    #properties = ["/location/statistical_region/population","/location/statistical_region/gdp_real","/location/statistical_region/cpi_inflation_rate"]
    properties = ["/location/statistical_region/cpi_inflation_rate"]
    #properties = ["/location/statistical_region/population"]
    
    # TODO: this function should now return the best parameters per relation 
    property2bestParams = OnePropertyMatrixFactorPredictor.crossValidate(trainMatrix, textMatrix, 4, properties, outputFileName, [learningRates, l2penalties, iterations, filterThresholds, learningRateBalances, scale])
    #predictor.runEval(trainMatrix, textMatrix, testMatrix, property2bestParams)