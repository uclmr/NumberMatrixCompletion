import abstractPredictor
import numpy
import heapq

class BaselinePredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # this keeps the average for each relation
        self.property2median = {}
        # this keeps the patterns for each relation
        # each pattern has a dict of regions and values associated with it 
        self.property2patterns = {}

    def predict(self, property, region):
        # collect all the values for this region found in related patterns
        values = []
        if property in self.property2patterns:
            patterns = self.property2patterns[property]
            for pattern, region2value in patterns.items():
                if region in region2value:
                    values.append(region2value[region])
                    print "region: ", region.encode('utf-8'), " pattern used: ", pattern.encode('utf-8'), " value: ", region2value[region]
        
        if len(values) > 0:
            return numpy.mean(values)
        else:
            return self.property2median[property]
            
    def train(self, trainMatrix, textMatrix, params=[False]):
        # whether we are scaling or not
        scaling = params[0]
        # if we are scaling, what is the scaling parameter?
        if scaling:
            scalingParam=float(params[1])
            print "Training with KLDE supprt scaling parameter: ", scalingParam
        else:
            print "Training without KLDE support scaling"
         
        # for each property, find the patterns that result in improving the average the most
        # it should get better initially as good patterns are added, but then down as worse ones are added
        for property, trainRegion2value in trainMatrix.items():
            print property, trainRegion2value
            # first get the average
            self.property2median[property] = numpy.median(trainRegion2value.values())
            self.property2patterns[property] = {}
            
            # this is used to store the msaes for each pattern
            patternKLDEs = []
            # we first need to rank all the  text patterns according to their msae
            for pattern, region2value in textMatrix.items():
                # make sure that it has at least two value in common with training data, otherwise we might get spurious stuff
                keysInCommon = list(set(region2value.keys()) & set(trainRegion2value.keys()))
                if len(keysInCommon) > 1:
                    klde = abstractPredictor.AbstractPredictor.KLDE(region2value, trainRegion2value)
                    if scaling:
                        # this is a version of KLDE scaled according to how many values were used in the calculation
                        # see Koren (2008), it is the opposite (lower is better) of the scaling factor in equation 2
                        scalingFactor = scalingParam/(scalingParam + len(keysInCommon))
                        klde *= scalingFactor
                    heapq.heappush(patternKLDEs, (klde, pattern))
            
            # now we have the patterns ordered according to their 
            
            # predict
            prediction = {}
            for region in trainRegion2value:
                prediction[region] = self.predict(property, region)
            # calculate the current score with 
            currentKLDE = self.KLDE(prediction, trainRegion2value)
            while True:
                # The pattern with the smallest KLDE is indexed at 0
                # the elememts are (KLDE, pattern) tuples
                klde, pattern = heapq.heappop(patternKLDEs)
                
                # add it to the classifiers
                self.property2patterns[property][pattern] = textMatrix[pattern]
                print "text pattern: " + pattern.encode('utf-8')
                print "KLDE:", klde
                print "MAPE:", abstractPredictor.AbstractPredictor.MAPE(textMatrix[pattern], trainRegion2value)
                print textMatrix[pattern]
                
                # predict
                for region in trainRegion2value:
                    prediction[region] = self.predict(property, region)
                
                # calculate new KLDE
                newKLDE = self.KLDE(prediction, trainRegion2value)
                print "KLDE of predictor before adding the pattern:", currentKLDE
                print "KLDE of predictor after adding the pattern:", newKLDE
                # if higher than before, remove the last pattern added and break
                if newKLDE > currentKLDE:
                    del self.property2patterns[property][pattern]
                    break
                else:
                    currentKLDE = newKLDE
            
                
          
if __name__ == "__main__":
    
    import sys
    import random
    
    baselinePredictor = BaselinePredictor()
    
    trainMatrix = baselinePredictor.loadMatrix(sys.argv[1])
    textMatrix = baselinePredictor.loadMatrix(sys.argv[2])
    testMatrix = baselinePredictor.loadMatrix(sys.argv[3])
    random.seed(13)
    bestParams = baselinePredictor.crossValidate(trainMatrix, textMatrix, 4, [[False],[True,0.125],[True,0.25],[True,0.5],[True,1],[True,2],[True,4]])
    baselinePredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)