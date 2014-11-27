import abstractPredictor
import numpy
import heapq

class FixedValuePredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # this keeps the median for each relation
        self.property2fixedValue = {}
        
        
    def predict(self, property, region):
        return self.property2fixedValue[property]
         
    
    def train(self, trainMatrix, textMatrix, params=None): 
        for property, trainRegion2value in trainMatrix.items():
            print property, trainRegion2value
            # try three options
            candidates = [0, numpy.median(trainRegion2value.values()), numpy.mean(trainRegion2value.values())]
            bestScore = float("inf")
            bestCandidate = None
            for candidate in candidates:    
                prediction = {}
                for region in trainRegion2value:
                    prediction[region] = candidate 
                mape = abstractPredictor.AbstractPredictor.MAPE(prediction, trainRegion2value)
                
                if mape < bestScore:
                    bestScore = mape
                    bestCandidate = candidate
                    
            if bestCandidate == 0:
                print property, " best value is 0 with score ", bestScore 
            elif bestCandidate == numpy.median(trainRegion2value.values()):
                print property, " best value is median with score ", bestScore
            elif bestCandidate == numpy.mean(trainRegion2value.values()):
                print property, " best value is mean with score ", bestScore                
            self.property2fixedValue[property] = bestCandidate
                            
                
          
if __name__ == "__main__":
    
    import sys

    
    fixedValuePredictor = FixedValuePredictor()
    
    trainMatrix = fixedValuePredictor.loadMatrix(sys.argv[1])
    textMatrix = fixedValuePredictor.loadMatrix(sys.argv[2])
    testMatrix = fixedValuePredictor.loadMatrix(sys.argv[3])

    bestParams = fixedValuePredictor.crossValidate(trainMatrix, textMatrix, 4)
    fixedValuePredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)