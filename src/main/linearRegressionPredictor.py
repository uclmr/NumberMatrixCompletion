import abstractPredictor
import numpy
from sklearn.preprocessing import Imputer
from sklearn.linear_model import ElasticNet

class LinearRegressionPredictor(abstractPredictor.AbstractPredictor):
    
    def __init__(self):
        # this keeps the regression model
        self.property2regressor = {}
        # this keeps the features for each region
        # this might include some kind of imputation. eg. with the mean? 
        self.imputedTextValueMatrix = None
        self.pattern2column = {}
        self.column2pattern = []
        self.region2row = {}
        self.property2median = {}
        
        
    def predict(self, property, region):
        # we first need to get the text features for that region
        if region in self.region2row:
            imputedTextFeatures = self.imputedTextValueMatrix[self.region2row[region]]
            return self.property2regressor[property].predict(imputedTextFeatures)
        else:
            print "No text patterns for region ", region, " returning the median for the property"
            return self.property2median[property]           
    
    def train(self, trainMatrix, textMatrix, params):
        # first we need to train the imputer and impute the training data
        
        # get the text data intp a matrix
        originalTextValueMatrix = []
        # initialize them to missing value lists using the entries of textMatrix:
        # This should have all the regions we have collected data for
        for region2value in textMatrix.values():
            for region in region2value:
                if region not in self.region2row:
                    self.region2row[region] = len(originalTextValueMatrix) 
                    originalTextValueMatrix.append(['NaN']*len(textMatrix))
        
        # add the knwon values            
        for patternNo, (pattern, region2value) in enumerate(textMatrix.items()):
            self.pattern2column[pattern] = patternNo
            self.column2pattern.append(pattern)          
            for region, value in region2value.items():
                originalTextValueMatrix[self.region2row[region]][patternNo] = value
        print "Fitting the data imputer"        
        textValuesImputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        #print originalTextValueMatrix
        #print pattern2column
        #print region2row
        self.imputedTextValueMatrix = textValuesImputer.fit_transform(originalTextValueMatrix)
        #print column2pattern[0:30]
        #print originalTextValueMatrix[0][0:30]
        #print imputedTextValueMatrix[0][0:30]
        #print originalTextValueMatrix[1][0:30]
        #print imputedTextValueMatrix[1][0:30]
        l1_ratio, l1_strength = params
        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property, " with params l1_ratio ", l1_ratio, " and l1_strength ", l1_strength
            # using the median of the property as back up
            self.property2median[property] = numpy.median(trainRegion2value.values())
            
            self.property2regressor[property] = ElasticNet(l1_strength, l1_ratio)
             
            # first construct the target values
            targetValues = []
            # occasionally we have missing values
            trainingVectors = []
            for region, value in trainRegion2value.items():
                targetValues.append(value)
                if region in self.region2row:
                    trainingVectors.append(self.imputedTextValueMatrix[self.region2row[region]])
                else:
                    print "No text patterns for region ", region, " skipping it in training"
            self.property2regressor[property].fit(trainingVectors, targetValues)
        
          
if __name__ == "__main__":
    
    import sys
    import random
    linearRegressionPredictor = LinearRegressionPredictor()
    
    trainMatrix = linearRegressionPredictor.loadMatrix(sys.argv[1])
    textMatrix = linearRegressionPredictor.loadMatrix(sys.argv[2])
    testMatrix = linearRegressionPredictor.loadMatrix(sys.argv[3])
    
    random.seed(13)
    bestParams = linearRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4 ,[[0.5,1.0]])
    linearRegressionPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)