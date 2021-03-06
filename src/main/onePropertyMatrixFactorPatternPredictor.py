import fixedValuePredictor
import onePropertyMatrixFactorPredictor
import numpy
import copy
import operator
from collections import Counter, OrderedDict
import operator
import scipy
import heapq 

class OnePropertyMatrixFactorPatternPredictor(onePropertyMatrixFactorPredictor.OnePropertyMatrixFactorPredictor):
    
    def __init__(self):
        # initialize the mf predictor, which should initialize the guess
        onePropertyMatrixFactorPredictor.OnePropertyMatrixFactorPredictor.__init__(self)
        # This is to store the patterns ultimately used for the pattern prediction
        self.property2patterns = {}
        
    # This is the same as the baseline predictor
    def predict(self, property, region, of, useDefault=True):
        # collect all the values for this region found in related patterns
        values = []
        if property in self.property2patterns:
            patterns = self.property2patterns[property]
            for pattern, region2value in patterns.items():
                if region in region2value:
                    values.append(region2value[region])
                    of.write("region: " + region.encode('utf-8') + " pattern used: " + pattern.encode('utf-8') + " value: " + str(region2value[region]) + "\n")

        if len(values) > 0:
            return numpy.mean(values)
        else:
            if useDefault:
                return fixedValuePredictor.FixedValuePredictor.predict(self, property, region, of)
            else:
                return None
    
    #@profile  
    def trainRelation(self, property, trainRegion2value, textMatrix, of, params):
                
        #of.write(str(trainRegion2value))
        # get the back up fixed values
        onePropertyMatrixFactorPredictor.OnePropertyMatrixFactorPredictor.trainRelation(self, property, trainRegion2value, textMatrix, of, params)

        # right so we got the vectors, we should now decide how many to keep
        self.property2patterns[property] = {}
        if property not in self.property2pattern2Vector:
            of.write(property + ", no patterns left after filtering, SKIP\n")
            return
        
        #print "OK"        
        patternDistances = []

        for pattern, vector in self.property2pattern2Vector[property].items():
            distance = (scipy.spatial.distance.cdist([self.property2vector[property]], [vector], 'euclidean'))[0][0]
        
            heapq.heappush(patternDistances, (distance, pattern))
            
        #print "OK"
        # predict
        prediction = {}
        for region in trainRegion2value:
            prediction[region] = self.predict(property, region, of)

        # calculate the current score with 
        currentMAPE = self.MAPE(prediction, trainRegion2value)
        
        while len(patternDistances) > 0:
            # The pattern with the smallest MAPE is indexed at 0
            # the elememts are (MAPE, pattern) tuples
            distance, pattern = heapq.heappop(patternDistances)
            
            # add it to the classifiers
            self.property2patterns[property][pattern] = textMatrix[pattern]
            
            of.write("text pattern: " + pattern.encode('utf-8') + "\n")
            
            of.write("distance:" + str(distance) + "\n")                
            #print "MASE", self.MASE(textMatrix[pattern], trainRegion2value)
            of.write(str(textMatrix[pattern]) + "\n")
            
            # predict
            prediction = {}
            
            for region in trainRegion2value:
                prediction[region] = self.predict(property, region, of)
            
            # calculate new MAPE
            newMAPE = self.MAPE(prediction, trainRegion2value)
            of.write("MAPE of predictor before adding the pattern:" + str(currentMAPE) + "\n")
            of.write("MAPE of predictor after adding the pattern:" + str(newMAPE) + "\n")
            # if higher than before, remove the last pattern added and break
            
            if newMAPE > currentMAPE:
                del self.property2patterns[property][pattern]
                break
            else:
                currentMAPE = newMAPE
            
        

                                     
if __name__ == "__main__":
    
    import sys
    import json
    import os.path
    
    # helps detect errors
    numpy.seterr(all='raise')
    #numpy.random.seed(13)
    
    
    predictor = OnePropertyMatrixFactorPatternPredictor()
    
    trainMatrix = predictor.loadMatrix(sys.argv[1])
    textMatrix = predictor.loadMatrix(sys.argv[2])
    testMatrix = predictor.loadMatrix(sys.argv[3])
    
    outputFileName = sys.argv[4]

    learningRates = [0.0001]
    l2penalties = [0.1]
    iterations =  [1000,2000, 3000, 4000,5000]
    filterThresholds = [0.5]
    learningRateBalances = [0.0]
    scale = [False]
    losses = ["SMAPE", "MAPE"] # , "AE", "SE", 

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
    properties = ["/location/statistical_region/" + sys.argv[5]]
    #properties = json.loads(open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/featuresKept.json").read())
    # removed the properties that have been done already.
    #properties.remove("/location/statistical_region/size_of_armed_forces")
    #properties.remove("/location/statistical_region/gni_per_capita_in_ppp_dollars")
    #properties.remove("/location/statistical_region/gdp_nominal")
    
    # Otherwise, specify which ones are needed:
    #properties = ["/location/statistical_region/population","/location/statistical_region/gdp_real","/location/statistical_region/cpi_inflation_rate"]
    #properties = ["/location/statistical_region/foreign_direct_investment_net_inflows"]
    #properties = ["/location/statistical_region/population"]
    #properties = ["/location/statistical_region/fertility_rate"]
    #properties = ["/location/statistical_region/trade_balance_as_percent_of_gdp"]
    #properties = ["/location/statistical_region/renewable_freshwater_per_capita"]
    #properties = ["/location/statistical_region/net_migration"]
    #properties = ["/location/statistical_region/gdp_growth_rate"]
 
    property2bestParams = OnePropertyMatrixFactorPatternPredictor.crossValidate(trainMatrix, textMatrix, 4, properties, outputFileName, paramSets)
    
#     property2bestParams = {"/location/statistical_region/fertility_rate": [0.0001, 0.01, 1000, 0.3, 1.0, True, 'SMAPE']}
    property2MAPE = {}
    for property in properties:
        paramsStrs = []
        for param in property2bestParams[property]:
            paramsStrs.append(str(param))
        
        ofn = outputFileName + "_" + property.split("/")[-1] + "_" + "_".join(paramsStrs) + "_TEST"
        a= {}
        OnePropertyMatrixFactorPatternPredictor.runRelEval(a, property, trainMatrix[property], textMatrix, testMatrix[property], ofn, property2bestParams[property])
        property2MAPE[property] = a.values()[0]
                        
    for property in sorted(property2MAPE):
        print property, property2MAPE[property]
    print "avg MAPE:", str(numpy.mean(property2MAPE.values()))

    
