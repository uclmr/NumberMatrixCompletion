'''
This is a baseline predictor. For each property, it finds the text patterns that correlate the best.
If the value for a country cannot be predicted in this way, it returns the average of the property
'''
import operator
import json
import numpy
import random
from sklearn.metrics import mean_squared_error
import math


class AbstractPredictor(object):
    def __init__(self):
        pass

    @staticmethod
    def loadMatrix(jsonFile):
        print "loading from file " + jsonFile
        with open(jsonFile) as freebaseFile:
            property2region2value = json.loads(freebaseFile.read())
        

        regions = set([])
        valueCounter = 0
        for property, region2value in property2region2value.items():
            # Check for nan values and remove them
            for region, value in region2value.items():
                if not numpy.isfinite(value):
                    del region2value[region]
                    print "REMOVED:", value, " for ", region, " ", property
            if len(region2value) == 0:
                del property2region2value[property]
                print "REMOVED property:", property, " no values left"
            else:
                valueCounter += len(region2value) 
                regions = regions.union(set(region2value.keys()))

        print len(property2region2value), " properties"
        print len(regions),  " unique regions"
        print valueCounter, " values loaded"
        return property2region2value
    
    def train(self, trainMatrix, textMatrix, params):
        pass
        
    def predict(self, property, region):
        pass
    
    @classmethod
    def runEval(cls, trainMatrix, textMatrix, testMatrix, property2params):
        predictor = cls()
        print "Training"
        predictor.train(trainMatrix, textMatrix, property2params)
        print "Testing"
        predMatrix = {}
        for property, region2value in testMatrix.items():
            print property
            predMatrix[property] = {}
            for region in region2value:
                predMatrix[property][region] = predictor.predict(property, region)
        
        avgScore = predictor.eval(predMatrix, testMatrix)
        return avgScore
     
    @classmethod
    def runRelEval(cls, property, trainRegion2value, textMatrix, testRegion2value, params):
        predictor = cls()
        print "Training"
        learningRate, regParam, iterations, filterThreshold, learningRateBalance = params
        predictor.trainRelation(property, trainRegion2value, textMatrix, learningRate, regParam, iterations, filterThreshold, learningRateBalance)
        print "Testing"
        predMatrix = {}
        predMatrix[property] = {}
        for region in testRegion2value:
            predMatrix[property][region] = predictor.predict(property, region)
        
        testMatrix = {}
        testMatrix[property] = testRegion2value
        avgScore = predictor.eval(predMatrix, testMatrix)
        return avgScore
    
    
    # the paramSets
    @classmethod
    def crossValidate(cls, trainMatrix, textMatrix, folds, properties, paramGroups):
        # first construct the folds per relation
        property2folds = {}
        # we set the same random in order to get the same folds every time
        # we do it on the whole dataset everytime independently of the choice of properties
        random.seed(13)
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
        
        # here we keep the best params found for each relation 
        property2bestParams = {}

        # for each of the properties we decide 
        for property in properties:
            print "property: " + property
            # this keeps the lowest MAPE achieved for this property across folds
            lowestAvgMAPE = float("inf")                


            #TODO: this is to handle the folds on different processors
            #mgr = multiprocessing.Manager()
            #d = mgr.dict()

            # for each parameter setting
            learningRates, l2penalties, iterations, filterThresholds, learningRateBalances = paramGroups
            # naive grid search
            paramSets = []
            for lr in learningRates:
                for l2 in l2penalties:
                    for iters in iterations:
                        for ft in filterThresholds:
                            for lrb in learningRateBalances:
                                paramSets.append([lr,l2,iters,ft,lrb])
                
            
            
            for params in paramSets:
                print "params: ", params 
                
                paramMAPEs = []
                # for each fold    
                
                for foldNo in xrange(folds):
                    print "fold:", foldNo
                    # construct the training and test datasets
                    foldTrainRegion2value = {}
                    foldTestRegion2value = {}
                    data = property2folds[property]
                    
                    foldTrainMatrix = {}
                    for idx in xrange(folds):
                        if (idx % folds) == foldNo:
                            # this the test data
                            foldTestRegion2value = data[idx]
                        else:
                            # the rest adds to the training data
                            foldTrainRegion2value.update(data[idx])
                            
                    # now create a predictor and run the eval
                    predictor = cls()
                    # run the eval
                    # TODO: this needs to be run for each relation now
                    mape = predictor.runRelEval(property, foldTrainRegion2value, textMatrix, foldTestRegion2value, params)
                    print "fold:", foldNo, " MAPE:", mape
                    # add the score for the fold
                    paramMAPEs.append(mape)
                    
                # get the average across folds    
                avgMAPE = numpy.mean(paramMAPEs)
                print "params:", params, " avgMAPE:", avgMAPE, "stdMAPE:", numpy.std(paramMAPEs), "foldMAPEs:", paramMAPEs
            
                # lower is better
                if avgMAPE < lowestAvgMAPE:
                    bestParams = params
                    lowestAvgMAPE = avgMAPE
                

            print property + ": lowestAvgMAPE:", lowestAvgMAPE
            print property + ": bestParams: ", bestParams
            property2bestParams[property] = bestParams
            
        # we return the best params 
        return bestParams
            
                
    @staticmethod
    def eval(predMatrix, testMatrix):
        print predMatrix
        print testMatrix
        property2MAPE = {}
        property2MASE = {}
        property2RMSE = {}
        for property, predRegion2value in predMatrix.items():
            print property
            #print "real: ", testMatrix[property]
            #print "predicted: ", predRegion2value
            mape = AbstractPredictor.MAPE(predRegion2value, testMatrix[property], True)
            print "MAPE: ", mape
            property2MAPE[property] = mape
            rmse = AbstractPredictor.RMSE(predRegion2value, testMatrix[property])
            print "RMSE: ", rmse
            property2RMSE[property] = rmse
            mase = AbstractPredictor.MASE(predRegion2value, testMatrix[property], True)
            print "MASE: ", mase
            property2MASE[property] = mase
            
        #return numpy.mean(MAPEs)
        print "properties ordered by MAPE"
        sortedMAPEs = sorted(property2MAPE.items(), key=operator.itemgetter(1))
        for property, mape in sortedMAPEs:
            print property, ":", mape 
                           
        print "properties ordered by MASE"
        sortedMASEs = sorted(property2MASE.items(), key=operator.itemgetter(1))
        for property, mase in sortedMASEs:
            print property, ":", mase 
        
        
        print "avg. MAPE: ", numpy.mean(property2MAPE.values())
        print "avg. RMSE: ", numpy.mean(property2RMSE.values())
        print "avg. MASE: ", numpy.mean(property2MASE.values())
        # we use MASE as the main metric, which is returned to guide the hyperparamter selection
        return numpy.mean(property2MAPE.values())
    
    # We follow the definitions of Chen and Yang (2004)
    # the second dict does the scaling
    # not defined when the trueDict value is 0
    # returns the mean absolute percentage error and the number of predicted values used in it
    @staticmethod
    def MAPE(predDict, trueDict, verbose=False):        
        absPercentageErrors = {}
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        
        
        # if there is a zero in the values we are evaluating against 
        #if 0.0 in trueDict.values():
        #    minAbsValue = float("inf")
        #    for value in trueDict.values():
        #        if numpy.abs(value) > 0 and numpy.abs(value) < minAbsValue:
        #            minAbsValue = numpy.abs(value)
        #else:
        #    minAbsValue = 0
        
        #print keysInCommon
        for key in keysInCommon:
            # avoid 0's
            if trueDict[key] != 0:
                absError = abs(predDict[key] - trueDict[key])
                absPercentageErrors[key] = absError/numpy.abs(trueDict[key])
        
        if len(absPercentageErrors) > 0:     
            if verbose:
                print "MAPE results"
                sortedAbsPercentageErrors = sorted(absPercentageErrors.items(), key=operator.itemgetter(1))
                print "top-5 predictions"
                print "region:pred:true"
                for idx in xrange(5):
                    print sortedAbsPercentageErrors[idx][0].encode('utf-8'), ":", predDict[sortedAbsPercentageErrors[idx][0]], ":", trueDict[sortedAbsPercentageErrors[idx][0]] 
                print "bottom-5 predictions"
                for idx in xrange(5):
                    print sortedAbsPercentageErrors[-idx-1][0].encode('utf-8'), ":", predDict[sortedAbsPercentageErrors[-idx-1][0]], ":", trueDict[sortedAbsPercentageErrors[-idx-1][0]]
            
            return numpy.mean(absPercentageErrors.values())
        else:
            return float("inf")

    
    # This is MASE, sort of proposed in Hyndman 2006
    # at the moment the evaluation metric of choice
    # it returns 1 if the method has the same absolute errors as the median of the test set.
    @staticmethod
    def MASE(predDict, trueDict, verbose=False):
        # first let's estimate the error from the median:
        median = numpy.median(trueDict.values())
        
        # calculate the errors of the test median
        # we are scaling with the error of the median on the value at question. This will be 0 often, thus we want to know the smallest non-zero to add it.
        minMedianAbsError = float("inf")
        for value in trueDict.values():
            medianAbsError = numpy.abs(value - median)
            if medianAbsError > 0 and medianAbsError < minMedianAbsError:
                minMedianAbsError = medianAbsError
        
    
        # get those that were predicted
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        predScaledAbsErrors = {}
        for key in keysInCommon:
            predScaledAbsErrors[key] = (numpy.abs(predDict[key] - trueDict[key]) + minMedianAbsError)/(numpy.abs(median - trueDict[key]) + minMedianAbsError)
        
        if verbose:
            print "MASE results"
            sortedPredScaledAbsErrors = sorted(predScaledAbsErrors.items(), key=operator.itemgetter(1))
            print "top-5 predictions"
            print "region:pred:true"
            for idx in xrange(5):
                print sortedPredScaledAbsErrors[idx][0].encode('utf-8'), ":", predDict[sortedPredScaledAbsErrors[idx][0]], ":", trueDict[sortedPredScaledAbsErrors[idx][0]] 
            print "bottom-5 predictions"
            for idx in xrange(5):
                print sortedPredScaledAbsErrors[-idx-1][0].encode('utf-8'), ":", predDict[sortedPredScaledAbsErrors[-idx-1][0]], ":", trueDict[sortedPredScaledAbsErrors[-idx-1][0]]
                
        return numpy.mean(predScaledAbsErrors.values())
                    

        

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
            print "KLDE results"
            sortedKLDEs = sorted(kldes.items(), key=operator.itemgetter(1))
            print "top-5 predictions"
            print "region:pred:true"
            for idx in xrange(5):
                print sortedKLDEs[idx][0].encode('utf-8'), ":", predDict[sortedKLDEs[idx][0]], ":", trueDict[sortedKLDEs[idx][0]] 
            print "bottom-5 predictions"
            for idx in xrange(5):
                print sortedKLDEs[-idx-1][0].encode('utf-8'), ":", predDict[sortedKLDEs[-idx-1][0]], ":", trueDict[sortedKLDEs[-idx-1][0]]
                
        return numpy.mean(kldes.values())
    
    # This does a scaling according to the number of values actually used in the calculation
    # The more values used, the lower the score (lower is better)
    # smaller scaling parameters make the number of values used more important, larger lead to the same as standard KLDE
    # Inspired by the shrunk correlation coefficient (Koren 2008 equation 2)
    @staticmethod
    def supportScaledKLDE(predDict, trueDict, scalingParam=1):
        klde = AbstractPredictor.KLDE(predDict, trueDict)
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))               
        scalingFactor = scalingParam/(scalingParam + len(keysInCommon))
        return klde * scalingFactor

    @staticmethod
    def supportScaledMASE(predDict, trueDict, scalingParam=1):
        mase = AbstractPredictor.MASE(predDict, trueDict)
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))               
        scalingFactor = float(scalingParam)/(scalingParam + len(keysInCommon))
        return mase * scalingFactor        

    @staticmethod
    def supportScaledMAPE(predDict, trueDict, scalingParam=1):
        mape = AbstractPredictor.MAPE(predDict, trueDict)
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))               
        scalingFactor = float(scalingParam)/(scalingParam + len(keysInCommon))
        return mape * scalingFactor        


    @staticmethod
    def RMSE(predDict, trueDict):
        keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))
        #print keysInCommon
        y_actual = []
        y_predicted = []
        for key in keysInCommon:
            y_actual.append(trueDict[key])
            y_predicted.append(predDict[key])
        return math.sqrt(mean_squared_error(y_actual, y_predicted))
        