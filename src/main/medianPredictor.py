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
        # for each property, find the patterns that result in improving the average the most
        # it should get better initially as good patterns are added, but then down as worse ones are added
        for property, trainRegion2value in trainMatrix.items():
            print property, trainRegion2value
            # first get the median
            self.property2median[property] = numpy.median(trainRegion2value.values())            
                
          
if __name__ == "__main__":
    
    import sys

    
    medianPredictor = MedianPredictor()
    
    trainMatrix = medianPredictor.loadMatrix(sys.argv[1])
    textMatrix = medianPredictor.loadMatrix(sys.argv[2])
    testMatrix = medianPredictor.loadMatrix(sys.argv[3])

    medianPredictor.crossValidate(trainMatrix, textMatrix)
    medianPredictor.runEval(trainMatrix, textMatrix, testMatrix, None)