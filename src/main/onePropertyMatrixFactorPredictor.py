import fixedValuePredictor
import numpy
import copy
import operator
from collections import Counter, OrderedDict
import operator
import scipy

class OnePropertyMatrixFactorPredictor(fixedValuePredictor.FixedValuePredictor):
    
    def __init__(self):
        # each region has a different vector for each property
        self.property2region2Vector = {}
        self.property2vector = {}
        self.property2pattern2Vector = {}
        # keep this as a backup:
        fixedValuePredictor.FixedValuePredictor.__init__(self)
        # This is the scaling factor
        self.scalingFactor = None
        
    def predict(self, property, region, of, useDefault=True):
        # it can be the case that we haven't got anything for a country
        if property in self.property2vector and region in self.property2region2Vector[property]:
            # remember to mulitply with the scaling factor
            return numpy.dot(self.property2vector[property], self.property2region2Vector[property][region]) * self.scalingFactor
        else:
            of.write("no vector for property " + property.encode('utf-8') + " or no vector for region " + region.encode('utf-8') + " for this property\n")
            if useDefault:
                return fixedValuePredictor.FixedValuePredictor.predict(self, property, region, of)
            else:
                return None
    
    #@profile  
    def trainRelation(self, property, trainRegion2value, textMatrix, of, params):
        
        learningRate, regParam, iterations, filterThreshold, learningRateBalance, scale, loss = params
        
        #of.write(str(trainRegion2value))
        # get the back up fixed values
        fixedValuePredictor.FixedValuePredictor.trainRelation(self, property, trainRegion2value, textMatrix, of)
                
        # first let's filter 
        filteredPatterns = []
        filteredPattern2MAPE = {}
        
        of.write(str(trainRegion2value))

        # this is to be used to avoid superhuge errors        
        #errorBound = max(numpy.abs(numpy.min(trainRegion2value.values())), numpy.abs(numpy.max(trainRegion2value.values())))
        errorBound = numpy.abs(numpy.median(trainRegion2value.values()))
        of.write("error bound:"  + str(errorBound) + "\n")
        scaling = scale
        # if scaling
        if scaling:
            #self.scalingFactor = max(numpy.abs(numpy.min(trainRegion2value.values())), numpy.abs(numpy.max(trainRegion2value.values())))/100
            self.scalingFactor = numpy.abs(numpy.median(trainRegion2value.values()))
            #[numpy.nonzero(trainRegion2value.values())]
            #self.scalingFactor = numpy.min(numpy.abs(trainRegion2value.values()))
            #self.scalingFactor = numpy.max(numpy.abs(trainRegion2value.values()))

            of.write("scaling factor:" + str(self.scalingFactor) + "\n")
            # scale the error bound too!
            errorBound /= self.scalingFactor
            of.write("error bound after scaling:"  + str(errorBound) + "\n")
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
                mape = self. MAPE(region2value, trainRegion2value)
                if mape < filterThreshold:
                    filteredPatterns.append(pattern)
                    filteredPattern2MAPE[pattern] = mape
                    # scale if necessary
                    if scaling:
                        scaledRegion2value = {}#copy.deepcopy(region2value)
                        for region, value in region2value.items():
                            if value != 0.0:
                                scaledRegion2value[region] = value/self.scalingFactor
                        scaledTextMatrix[pattern] = scaledRegion2value
                    else:
                        scaledTextMatrix[pattern] = region2value
                    # order it so that we don't have issues with random init           
                    scaledTextMatrix[pattern] = OrderedDict(sorted(scaledTextMatrix[pattern].items(), key=lambda t: t[0]))
        of.write(property+ ", " + str(len(filteredPatterns)) +" patterns left after filtering\n")
        for pattern in filteredPatterns:
            of.write(pattern.encode('utf-8') + "\t" + str(textMatrix[pattern])+"\n")
            
        # print the patterns ordered by value
        sortedPatterns = sorted(filteredPattern2MAPE.items(), key=operator.itemgetter(1))
        for p in sortedPatterns:
            of.write(str(p) + "\n")
            
        #of.write(str(filteredPatterns).encode('utf-8') +"\n")
        if len(filteredPatterns) == 0:
            of.write(property + ", no patterns left after filtering, SKIP\n")
            return
        
        # if scaling
        if scaling:
            # replace (without over-writing) the original training values
            scaledTrainRegion2value = {}
            for region, value in trainRegion2value.items():
                if value != 0.0:
                    scaledTrainRegion2value[region] = value/self.scalingFactor            
        else:
            scaledTrainRegion2value = trainRegion2value
        
        # order it so that we don't have issues with random init 
        scaledTrainRegion2value = OrderedDict(sorted(scaledTrainRegion2value.items(), key=lambda t: t[0]))
        #of.write(str(scaledTrainRegion2value))

        
        of.write(property + " training starting now\n")

        # set it according to the text patterns   
        dims = max(2, int(numpy.ceil(numpy.sqrt(len(filteredPatterns)))))
        of.write(property + ", set the dimensions to the square root of the text patterns = " + str(dims) + "\n") 
    
        #regParam /= numpy.power(dims, 0.1)
        #print property, "set the reg param to ", regParam

        # set the random seed for reproducibility
        #numpy.random.seed(13)
        prng = numpy.random.RandomState(13)
    
        # initialize the low dim representations
        # first the property
        propertyVector = prng.rand(dims)/100
        

        # then the patterns and the regions
        region2Vector = {}            
        pattern2vector = {}
        # also count the times they appear in the patterns
        trainingRegion2counts = Counter()
        valuesPresent = 0
        
        regV = prng.rand(dims)/100
        
        for pattern in filteredPatterns:
            pattern2vector[pattern] = prng.rand(dims)/100
            valuesPresent += len(scaledTextMatrix[pattern]) 
            for region in scaledTextMatrix[pattern].keys():
                if region not in region2Vector:
                    region2Vector[region] =  copy.deepcopy(regV)
                if region in scaledTrainRegion2value:
                    trainingRegion2counts[region] += 1
                    
        for region in scaledTrainRegion2value.keys():
            if region not in region2Vector:
                del scaledTrainRegion2value[region]
                
                    
        
        of.write(property + ", regions after filtering: " + str(len(region2Vector)) + "\n")
        
        of.write(property + ", values present " + str(valuesPresent) + " density " + str(float(valuesPresent)/(len(filteredPatterns)*len(region2Vector))) + "\n")
                
        allpps = [property] + filteredPatterns
        
        # prepare the data for the MF:
        pp2scaledRegVals = {}
        pp2scaledRegVals[property] = scaledTrainRegion2value.items()
        for pattern in filteredPatterns:
            pp2scaledRegVals[pattern] = scaledTextMatrix[pattern].items()
        

        # calculate the initial trainError just to make sure we have the same starting point
        absoluteErrors = []
        for region, value in trainRegion2value.items():
            if region in region2Vector:
                # remember to multiply with the scaling factor
                pred = numpy.dot(propertyVector,region2Vector[region]) * self.scalingFactor
                error = pred - value 
                absoluteErrors.append(numpy.absolute(error))
        of.write(property + ", initial reconstruction mean absolute error on trainMatrix=" + str(numpy.mean(absoluteErrors)) + "\n") 
        
        #of.write(str(scaledTextMatrix))
        for iter in xrange(iterations):
            prng.shuffle(allpps)            
            for pp, regVals in pp2scaledRegVals.items():
                # we might be getting the values from either the train matrix or the 
                if pp == property:
                    #region2value = scaledTrainRegion2value
                    ppVector = propertyVector
                    # +1 is for the cases where we haven't seen this training region with any pattern
                    lr = (learningRateBalance*trainingRegion2counts[region] + 1) * learningRate                    
                else:
                    #region2value = scaledTextMatrix[pp]
                    ppVector = pattern2vector[pp]
                    lr = learningRate
                    
                # let's try to reconstruct each known value    
                #regVals = region2value.items()
                prng.shuffle(regVals)
                for region, value in regVals:
                    # we might not have a vector for this region, so ignore
                    #if value != 0.0:
                        # reconstruction error
                        # so this the squared percentage error loss (the denominator is a squared constant)
                        #if pp == property:
                    if loss == "SE":
                        eij = (value - numpy.dot(ppVector,region2Vector[region]))#/numpy.square(value)
                    elif loss == "SMAPE":
                        eij = (value - numpy.dot(ppVector,region2Vector[region]))/numpy.square(value)
                    elif loss == "MAPE":
                        eij = numpy.sign(value - numpy.dot(ppVector,region2Vector[region]))
                        #else:
                        #eij = (value - numpy.dot(ppVector,region2Vector[region]))
                            
                        #of.write(region.encode('utf-8') + " " + pp.encode('utf-8') + " original value:" + str(value) + " error:" + str(eij) + "\n")                            
                        # if the error is too large (2 times the max abs value) then set it to that
                    if abs(eij) > errorBound:                                
                        eij = errorBound * numpy.sign(eij)
                            
                        # if the error is very big (more than the square of the original value)
                        # just update straight
                        #if numpy.abs(eij) > 1:                                
                        #    eij = numpy.sign(eij)                            
         
                    ppVector += lr * (2 * eij * region2Vector[region] - regParam * ppVector)
                    region2Vector[region] += lr * (2 * eij * ppVector - regParam * region2Vector[region])
        
            # let's calculate the squared reconstruction error
            # maybe look only at the training data?
            #squaredErrors = []
            if iter % 100 == 0:
                absoluteErrors = []
                preds = {}
                for region, value in trainRegion2value.items():
                    if region in region2Vector:
                        pred = numpy.dot(propertyVector,region2Vector[region]) * self.scalingFactor
                        error = pred - value
                        absoluteErrors.append(numpy.absolute(error))
                        #squaredErrors.append(numpy.square(error))
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
                        if value != 0.0:
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
                
                distanceFromPropertyVector = {}
                for pattern, vector in pattern2vector.items():
                    distanceFromPropertyVector[pattern] = (scipy.spatial.distance.cdist([propertyVector], [vector], 'euclidean'))[0][0]
                
                sortedPaterns= sorted(distanceFromPropertyVector.items(), key=operator.itemgetter(1))
                
                of.write("top patterns closest to the property : distance\n")
                for idx in xrange(min(30, len(sortedPaterns))):
                    of.write(sortedPaterns[idx][0].encode('utf-8') + ":" +  str(sortedPaterns[idx][1])+ "\n")
                
                of.write("bottom patterns further from the property : distance\n")
                for idx in xrange(min(30, len(sortedPaterns))):
                    of.write(sortedPaterns[-idx-1][0].encode('utf-8') + ":" +  str(sortedPaterns[-idx-1][1])+ "\n")
                
            if mape < 0.000001:
                break
        
        distanceFromPropertyVector = {}
        for pattern, vector in pattern2vector.items():
            distanceFromPropertyVector[pattern] = (scipy.spatial.distance.cdist([propertyVector], [vector], 'euclidean'))[0][0]
                
        sortedPaterns= sorted(distanceFromPropertyVector.items(), key=operator.itemgetter(1))
                
        of.write("patterns sorted by distance from the property : distance\n")
        for idx in xrange(len(sortedPaterns)):
            of.write(sortedPaterns[idx][0].encode('utf-8') + ":" +  str(sortedPaterns[idx][1])+ "\n")

        
        
        #d[property] = (propertyVector, region2Vector)
        self.property2vector[property] = propertyVector
        self.property2region2Vector[property] = region2Vector 
        
        self.property2pattern2Vector[property] = pattern2vector
        
                                     
if __name__ == "__main__":
    
    import sys
    import os.path
    import json
    # helps detect errors
    numpy.seterr(all='raise')
    #numpy.random.seed(13)
    
    
    predictor = OnePropertyMatrixFactorPredictor()
    
    trainMatrix = predictor.loadMatrix(sys.argv[1])
    textMatrix = predictor.loadMatrix(sys.argv[2])
    testMatrix = predictor.loadMatrix(sys.argv[3])
    
    outputFileName = sys.argv[4]

    learningRates = [0.00001, 0.0001, 0.001]
    l2penalties = [0.1, 0.01]
    iterations =  [1000,2000,3000]
    filterThresholds = [0.1, 0.2, 0.3]
    learningRateBalances = [0.0, 1.0, 2.0]
    scale = [True]
    losses = ["SMAPE", "SE"] # , "SE", "SMAPE", "MAPE"

    # construct the grid for paramsearch:
    # naive grid search
    paramSets = []
    for lr in learningRates:
        for l2 in l2penalties:
            for iters in iterations:
                for ft in filterThresholds:
                    for lrb in learningRateBalances:
                        for sc in scale:
                            for loss in losses:
                                paramSets.append([lr,l2,iters,ft,lrb, sc, loss])
                            
    #paramSets = [[0.0001, 0.2, 20000, 0.012, 1.0, True]]
    
    # These are the winning ones:
    #learningRates = [0.0001]
    #l2penalties = [0.3]
    #iterations = [10000]
    #filterThresholds = [0.012]
    #learningRateBalances = [0.0, 1.0]
    
    # this loads all relations
    #properties = json.loads(open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/featuresKept.json").read())
    properties = ["/location/statistical_region/" + sys.argv[5]]
    # Otherwise, specify which ones are needed:
    #properties = ["/location/statistical_region/population","/location/statistical_region/gdp_real","/location/statistical_region/cpi_inflation_rate"]
    #properties = ["/location/statistical_region/cpi_inflation_rate"]
    #properties = ["/location/statistical_region/population"]
    #properties = ["/location/statistical_region/fertility_rate"]
    #properties = ["/location/statistical_region/trade_balance_as_percent_of_gdp"]
    #properties = ["/location/statistical_region/renewable_freshwater_per_capita"]
    #properties = ["/location/statistical_region/net_migration"]
    #properties = ["/location/statistical_region/gdp_growth_rate"]
 
    property2bestParams = OnePropertyMatrixFactorPredictor.crossValidate(trainMatrix, textMatrix, 4, properties, outputFileName, paramSets)

    #property2bestParams = {"/location/statistical_region/population": [5e-05, 0.05, 6000, 0.4, 0.5, True]}
    property2MAPE = {}
    for property in properties:
        paramsStrs = []
        for param in property2bestParams[property]:
            paramsStrs.append(str(param))
      
        ofn = outputFileName + "_" + property.split("/")[-1] + "_" + "_".join(paramsStrs) + "_TEST"
        a= {}
        OnePropertyMatrixFactorPredictor.runRelEval(a, property, trainMatrix[property], textMatrix, testMatrix[property], ofn, property2bestParams[property])
        property2MAPE[property] = a.values()[0]
                      
    for property in sorted(property2MAPE):
        print property, property2MAPE[property]
    print "avg MAPE:", str(numpy.mean(property2MAPE.values()))

    
