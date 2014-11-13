import abstractPredictor
import numpy
from astropy.wcs.docstrings import dims

class MatrixFactorPredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        self.region2Vector = {}
        self.propertyOrPattern2vector = {}
        
    def predict(self, property, region):
        return numpy.dot(self.property2vector[property], self.region2Vector[region])
    
    # parameters are: dimensions of vectors, learning rate, reg_parameter, iterations
    def train(self, trainMatrix, textMatrix, params=[10,0.1,1, 5000]):
        
        dims, learningRate, regParam, iterations = params
        # possibly filter to avoid features with low counts
        
        # initialize the low dim representations
        # first the properties
        for property, region2value in trainMatrix.items():
            self.propertyOrPattern2vector[property] = numpy.random.rand(dims)
            for region in region2value.keys():
                if region not in self.region2Vector:
                    self.region2Vector[region] = numpy.random.rand(dims)

        for pattern, region2value in textMatrix.items():
            self.propertyOrPattern2vector[pattern] = numpy.random.rand(dims)
            for region in region2value.keys():
                if region not in self.region2Vector:
                    self.region2Vector[region] = numpy.random.rand(dims)                    
                     
        
        # let's go!
        for iter in xrange(iterations):
            # for each property or pattern
            for pp in trainMatrix.keys() + textMatrix.keys():
                # we might be getting the values from either the train matrix or the 
                if pp in trainMatrix:
                    region2value = trainMatrix[pp]
                else:
                    region2value = textMatrix[pp]
                # let's try to reconstruct each known value    
                for region, value in region2value.items():
                    # reconstruction error
                    eij = value - numpy.dot(self.propertyOrPattern2vector[pp],self.region2Vector[region])
                    print pp, " ", region, " error=",eij
                    for k in xrange(dims):
                        self.propertyOrPattern2vector[pp][k] += learningRate * (2 * eij * self.region2Vector[region][k] - regParam * self.propertyOrPattern2vector[pp][k])
                        self.region2Vector[region][k] += learningRate * (2 * eij * self.propertyOrPattern2vector[pp][k] - regParam * self.region2Vector[region][k])        
                
        
            # let's calculate the squared reconstruction error
            # maybe look only at the training data?
            re = 0
            for property, region2value in trainMatrix.items():
                for region, value in region2value.items():
                    pred = self.predict(property, region)
                    re += numpy.square(pred - value)
             
            print "Iteration ", iter, " reconstruction RMSE on trainMatrix=", numpy.sqrt(re)  
if __name__ == "__main__":
    
    import sys
    # helps detect errors
    numpy.seterr(all='raise')
    # set the random seed for reproducibility
    numpy.random.seed(13)
    
    baselinePredictor = MatrixFactorPredictor()
    
    trainMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[1])
    textMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[2])
    testMatrix = abstractPredictor.AbstractPredictor.loadMatrix(sys.argv[3])

    bestParams = MatrixFactorPredictor.crossValidate(trainMatrix, textMatrix, 4, [[10, 0.0001,1, 5000]])
    #MatrixFactorPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)