import fixedValuePredictor
import numpy
import heapq

class BaselinePredictor(fixedValuePredictor.FixedValuePredictor):
    
    def __init__(self):
        # this keeps the patterns for each relation
        # each pattern has a dict of regions and values associated with it 
        self.property2patterns = {}
        # this initializes the fixed value
        fixedValuePredictor.FixedValuePredictor.__init__(self)
        #super(BaselinePredictor,self).__init_()
        

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
            return fixedValuePredictor.FixedValuePredictor.predict(self, property, region) 
            
    def train(self, trainMatrix, textMatrix, params=[False]):
        fixedValuePredictor.FixedValuePredictor.train(self, trainMatrix, textMatrix)
        
        # whether we are scaling or not
        scaling = params[0]
        # if we are scaling, what is the scaling parameter?
        if scaling:
            scalingParam=float(params[1])
            print "Training with MAPE supprt scaling parameter: ", scalingParam
        else:
            print "Training without MAPE support scaling"
         
        # for each property, find the patterns that result in improving the average the most
        # it should get better initially as good patterns are added, but then down as worse ones are added
        for property, trainRegion2value in trainMatrix.items():
            print property, trainRegion2value
            # first get the average
            #self.property2median[property] = numpy.median(trainRegion2value.values())
            self.property2patterns[property] = {}
            
            # this is used to store the msaes for each pattern
            patternMAPEs = []
            # we first need to rank all the  text patterns according to their msae
            for pattern, region2value in textMatrix.items():
                # make sure that it has at least two value in common with training data, otherwise we might get spurious stuff
                keysInCommon = list(set(region2value.keys()) & set(trainRegion2value.keys()))
                if len(keysInCommon) > 1:
                    if scaling:
                        mape = self.supportScaledMAPE(region2value, trainRegion2value, scalingParam)                    
                    else:
                        mape = self.MAPE(region2value, trainRegion2value)
                    heapq.heappush(patternMAPEs, (mape, pattern))
            
            # predict
            prediction = {}
            for region in trainRegion2value:
                prediction[region] = self.predict(property, region)
            # calculate the current score with 
            currentMAPE = self.MAPE(prediction, trainRegion2value)
            while True:
                # The pattern with the smallest MAPE is indexed at 0
                # the elememts are (MAPE, pattern) tuples
                mape, pattern = heapq.heappop(patternMAPEs)
                
                # add it to the classifiers
                self.property2patterns[property][pattern] = textMatrix[pattern]
                print "text pattern: " + pattern.encode('utf-8')
                print "MAPE:", mape                
                print "MASE", self.MASE(textMatrix[pattern], trainRegion2value)
                print textMatrix[pattern]
                
                # predict
                for region in trainRegion2value:
                    prediction[region] = self.predict(property, region)
                
                # calculate new MAPE
                newMAPE = self.MAPE(prediction, trainRegion2value)
                print "MAPE of predictor before adding the pattern:", currentMAPE
                print "MAPE of predictor after adding the pattern:", newMAPE
                # if higher than before, remove the last pattern added and break
                if newMAPE > currentMAPE:
                    del self.property2patterns[property][pattern]
                    break
                else:
                    currentMAPE = newMAPE
            
                
          
if __name__ == "__main__":
    
    import sys
    
    baselinePredictor = BaselinePredictor()
    
    trainMatrix = baselinePredictor.loadMatrix(sys.argv[1])
    textMatrix = baselinePredictor.loadMatrix(sys.argv[2])
    testMatrix = baselinePredictor.loadMatrix(sys.argv[3])

    bestParams = baselinePredictor.crossValidate(trainMatrix, textMatrix, 4, [[False],[True, 0.03125],[True, 0.0625],[True, 0.125],[True, 0.25],[True,0.5],[True,1],[True,2],[True,4],[True,8],])
    baselinePredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)