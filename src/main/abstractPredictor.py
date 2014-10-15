'''
This is a baseline predictor. For each property, it finds the text patterns that correlate the best.
If the value for a country cannot be predicted in this way, it returns the average of the property
'''
import operator
import json
import numpy
import random

class AbstractPredictor(object):
    def __init__(self):
        pass

    @staticmethod
    def loadMatrix(jsonFile):
        print "loading from file " + jsonFile
        with open(jsonFile) as freebaseFile:
            property2region2value = json.loads(freebaseFile.read())
        
        print len(property2region2value), " properties"
        regions = set([])
        valueCounter = 0
        for region2value in property2region2value.values():
            valueCounter += len(region2value) 
            regions = regions.union(set(region2value.keys()))
        print len(regions),  " unique regions"
        print valueCounter, " values loaded"
        return property2region2value
    
    def train(self, trainMatrix, textMatrix, params):
        pass
        
    def predict(self, property, region):
        pass
    
    @classmethod
    def runEval(cls, trainMatrix, textMatrix, testMatrix, params):
        predictor = cls()
        print "Training"
        predictor.train(trainMatrix, textMatrix, params)
        print "Testing"
        predMatrix = {}
        for property, region2value in testMatrix.items():
            print property
            predMatrix[property] = {}
            for region in region2value:
                predMatrix[property][region] = predictor.predict(property, region)
        
        avgKLDE = predictor.eval(predMatrix, testMatrix)
        return avgKLDE
    
    
    # the params vector is the set of parameters to try out
    @classmethod
    def crossValidate(cls, trainMatrix, textMatrix, folds=4, paramSets=[None]):
        # first construct the folds per relation
        property2folds = {}
        # For each property
        for property, region2value in trainMatrix.items():
            # create the empty folds
            property2folds[property] = [{} for _ in xrange(folds)]
            # shuffle the data points
            regions = region2value.keys()
            random.shuffle(regions)
            for idx, region in enumerate(regions):
                # pick a fold
                foldNo = idx % folds
                # add the datapoint there
                property2folds[property][foldNo][region] = region2value[region]
        
        bestParams = None
        lowestAvgKLDE = float("inf")

        # for each parameter setting
        for params in paramSets:
            paramKLDEs = []
            # for each fold    
            for foldNo in xrange(folds):
                print "fold:", foldNo
                # construct the training and test datasets
                foldTrainMatrix = {}
                foldTestMatrix = {}
                # for each property
                for property, data in property2folds.items():
                    foldTrainMatrix[property] = {}
                    for idx in xrange(folds):
                        if (idx % folds) == foldNo:
                            # this the test data
                            foldTestMatrix[property] = data[idx]
                        else:
                            # the rest adds to the training data
                            foldTrainMatrix[property].update(data[idx])
                # now create a predictor and run the eval
                predictor = cls()
                # run the eval
                klde = predictor.runEval(foldTrainMatrix, textMatrix, foldTestMatrix, params)
                print "fold:", foldNo, " KLDE:", klde
                # add the score for the fold
                paramKLDEs.append(klde)
            # get the average across folds    
            avgKLDE = numpy.mean(paramKLDEs)
            print "params:", params, " avgKLDE:", avgKLDE, "stdKLDE:", numpy.std(paramKLDEs), "foldKLDEs:", paramKLDEs
            
            # lower is better
            if avgKLDE < lowestAvgKLDE:
                bestParams = params
                lowestAvgKLDE = avgKLDE

        print "lowestAvgKLDE:", lowestAvgKLDE

        # we return the best params 
        return bestParams
            
                
    @staticmethod
    def eval(predMatrix, testMatrix):
        property2MAPE = {}
        property2KLDE = {}
        for property, predRegion2value in predMatrix.items():
            print property
            #print "real: ", testMatrix[property]
            #print "predicted: ", predRegion2value
            mape = AbstractPredictor.MAPE(predRegion2value, testMatrix[property])
            print "MAPE: ", mape
            property2MAPE[property] = mape
            klde = AbstractPredictor.KLDE(predRegion2value, testMatrix[property], True)
            print "KLDE: ", klde
            property2KLDE[property] = klde
        #return numpy.mean(MAPEs)
        print "properties ordered by MAPE"
        sortedMAPEs = sorted(property2MAPE.items(), key=operator.itemgetter(1))
        for property, mape in sortedMAPEs:
            print property, ":", mape 
               
        print "properties ordered by KLDE"
        sortedKLDEs = sorted(property2KLDE.items(), key=operator.itemgetter(1))
        for property, klde in sortedKLDEs:
            print property, ":", klde 
        
        print "avg. MAPE: ", numpy.mean(property2MAPE.values())
        print "avg. KLDE: ", numpy.mean(property2KLDE.values())
        return numpy.mean(property2KLDE.values())
    
    # We follow the definitions of Chen and Yang (2004)
    # the second dict does the scaling
    # not defined when the trueDict value is 0
    # returns the mean absolute percentage error and the number of predicted values used in it
    @staticmethod
    def MAPE(predDict, trueDict):
        absPercentageErrors = []
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        #print keysInCommon
        for key in keysInCommon:
            if trueDict[key] != 0:
                absError = abs(predDict[key] - trueDict[key])
                absPercentageErrors.append(absError/abs(trueDict[key]))
        #print absPercentageErrors
        if len(absPercentageErrors) > 0:    
            return numpy.mean(absPercentageErrors)
        else:
            return "undefined"

    # This is the KL-DE1 measure defined in Chen and Yang (2004)        
    @staticmethod
    def KLDE(predDict, trueDict, verbose=False):
        kldes = {}
        # first we need to get the stdev used in scaling
        # let's use all the values for this, not only the ones in common
        std = numpy.std(trueDict.values())
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        
        for key in keysInCommon:
            scaledAbsError = abs(predDict[key] - trueDict[key])/std
            klde = numpy.exp(-scaledAbsError) + scaledAbsError - 1
            kldes[key] = klde
        
        if verbose:
            sortedKLDEs = sorted(kldes.items(), key=operator.itemgetter(1))
            print "top-5 predictions"
            print "region:pred:true"
            for idx in xrange(5):
                print sortedKLDEs[idx][0].encode('utf-8'), ":", predDict[sortedKLDEs[idx][0]], ":", trueDict[sortedKLDEs[idx][0]] 
            print "bottom-5 predictions"
            for idx in xrange(5):
                print sortedKLDEs[-idx-1][0].encode('utf-8'), ":", predDict[sortedKLDEs[-idx-1][0]], ":", trueDict[sortedKLDEs[-idx-1][0]]
                
        return numpy.mean(kldes.values())        
    
    