'''
This is a baseline predictor. For each property, it finds the text patterns that correlate the best.
If the value for a country cannot be predicted in this way, it returns the average of the property
'''

import json
import numpy

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
            klde = AbstractPredictor.KLDE(predRegion2value, testMatrix[property])
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
        
        for key in keysInCommon:
            if trueDict[key] != 0:
                absError = abs(predDict[key] - trueDict[key])
                absPercentageErrors.append(absError/abs(trueDict[key]))
            
        return numpy.mean(absPercentageErrors)

    # This is the KL-DE1 measure defined in Chen and Yang (2004)        
    @staticmethod
    def KLDE(predDict, trueDict):
        kldes = []
        # first we need to get the stdev used in scaling
        # let's use all the values for this, not only the ones in common
        std = numpy.std(trueDict.values())
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        
        for key in keysInCommon:
            scaledAbsError = abs(predDict[key] - trueDict[key])/std
            klde = numpy.exp(-scaledAbsError) + scaledAbsError - 1
            kldes.append(klde)
        return numpy.mean(kldes)        