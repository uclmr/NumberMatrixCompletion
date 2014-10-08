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
        predictor.train(trainMatrix, textMatrix, params)
        
        predMatrix = {}
        for property, region2value in testMatrix.items():
            predMatrix[property] = {}
            for region in region2value:
                predMatrix[property][region] = predictor.predict(property, region)
        
        avgKLDE = predictor.eval(predMatrix, testMatrix)
        return avgKLDE
    
    
    # the params vector is the set of parameters to try out
    @classmethod
    def crossValidate(cls, trainMatrix, textMatrix, folds=4, params=[None]):
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
                property2folds[property][foldNo][region] = region2value[region]
                
            
            
        for foldNo in xrange(folds):
            foldTrainMatrix = {}
            foldTestMatrix = {}
            for property, data in property2folds.items():
                for i,  in enumerate(instances):
                    if (i % folds) == fold:
                        testingInstances.append(inst)
                        testingInstancesIndices.append(i)
                    else:
                        trainingInstances.append(inst)

      
        # we return the best params 
        return bestParams
            
                
    @staticmethod
    def eval(predMatrix, testMatrix):
        MAPEs = []
        KLDEs = []
        for property, predRegion2value in predMatrix.items():
            print property
            #print "real: ", testMatrix[property]
            #print "predicted: ", predRegion2value
            mape = AbstractPredictor.MAPE(predRegion2value, testMatrix[property])
            print "MAPE: ", mape
            MAPEs.append(mape)
            klde = AbstractPredictor.KLDE(predRegion2value, testMatrix[property], True)
            print "KLDE: ", klde
            KLDEs.append(klde)
        #return numpy.mean(MAPEs)
        print "avg. MAPE: ", numpy.mean(MAPEs)
        print "avg. KLDE: ", numpy.mean(KLDEs)
        return numpy.mean(KLDEs)
    
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
    
    