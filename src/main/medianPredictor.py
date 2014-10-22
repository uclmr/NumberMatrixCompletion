import abstractPredictor
import numpy
import heapq

class MedianPredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # this keeps the median for each relation
        self.property2median = {}
        
        
    def predict(self, property, region):
        return self.property2median[property]
         
    
    def train(self, trainMatrix, textMatrix, params): 
        for property, trainRegion2value in trainMatrix.items():
            print property, trainRegion2value
            if params[0] == "median":
                self.property2median[property] = numpy.median(trainRegion2value.values())
            elif params[0] == "mean":
                self.property2median[property] = numpy.mean(trainRegion2value.values())
                            
                
          
if __name__ == "__main__":
    
    import sys

    
    medianPredictor = MedianPredictor()
    
    trainMatrix = medianPredictor.loadMatrix(sys.argv[1])
    textMatrix = medianPredictor.loadMatrix(sys.argv[2])
    testMatrix = medianPredictor.loadMatrix(sys.argv[3])

    bestParams = medianPredictor.crossValidate(trainMatrix, textMatrix, 4, [["median"],["mean"]])
    medianPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)