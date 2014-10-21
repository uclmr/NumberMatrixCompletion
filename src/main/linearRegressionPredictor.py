import abstractPredictor
import numpy
import scipy
from sklearn.preprocessing import Imputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from copy import copy
import operator

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
        # This indicates whether we use the regressor to emulate the neighborhood style model of Koren (2008), eq 6.
        self.neighborhoodModel = False
        
        
    def predict(self, property, region):
        # we first need to get the text features for that region
        if region in self.region2row:# and property =="/location/statistical_region/population":
            imputedTextFeatures = self.imputedTextValueMatrix[self.region2row[region]]
            # if we are considering the neighborhood model then we need to add the median back
            if self.neighborhoodModel:
                return self.property2median[property] + self.property2regressor[property].predict(imputedTextFeatures)
            else:
                return self.property2regressor[property].predict(imputedTextFeatures)
        else:
            print "No text patterns for region ", region.encode('utf-8'), " returning the median for the property"
            return self.property2median[property]           
    
    def train(self, trainMatrix, textMatrix, params):
        # first we need to train the imputer and impute the training data
        minCountries = params[0]
        # set the parameter whether we are doing the neighborhood model 
        self.neighborhoodModel = params[1]
        
        # get the text data into a matrix
        originalTextValueMatrix = []
        
        # let's get rid of all the text features that do not have values for at least two countries
        # in the training data
        trainingCountries = set([])
        for region2value in trainMatrix.values():
            trainingCountries.update(region2value.keys())
        
        # keep here the filtered text matrix using only the features
        # appearing at least minCountries times in the training
        print "Original text features: ", len(textMatrix)
        filteredTextMatrix = {}
        for pattern, region2value in textMatrix.items():
            if len(set(region2value.keys()) & trainingCountries) >= minCountries:# and "population" in pattern:
                filteredTextMatrix[pattern] = copy(region2value)
                
        print "Text features after removing those less than ", minCountries, " occurrences:", len(filteredTextMatrix)
        #print filteredTextMatrix
        
        # initialize them to missing value lists using the entries of textMatrix:
        # This should have all the regions we have collected data for
        for region2value in filteredTextMatrix.values():
            for region in region2value:
                if region not in self.region2row:
                    self.region2row[region] = len(originalTextValueMatrix)
                    # if we are doing the neighborhood version, unseen values are zeros
                    if self.neighborhoodModel:
                        originalTextValueMatrix.append(scipy.zeros(len(filteredTextMatrix)))
                    else: 
                        originalTextValueMatrix.append(['NaN']*len(filteredTextMatrix))
        
        # add the known values            
        for patternNo, (pattern, region2value) in enumerate(filteredTextMatrix.items()):
            # if we are doing neighborhood, get the median
            if self.neighborhoodModel:
                patternMedian = numpy.median(region2value.values())
            self.pattern2column[pattern] = patternNo
            self.column2pattern.append(pattern)          
            for region, value in region2value.items():
                # if neighborhood, subtract the median
                if self.neighborhoodModel:
                    originalTextValueMatrix[self.region2row[region]][patternNo] = value - patternMedian
                else:
                    originalTextValueMatrix[self.region2row[region]][patternNo] = value
        print numpy.shape(originalTextValueMatrix)
        # if neighborhood, we are done
        if self.neighborhoodModel:
            self.imputedTextValueMatrix =  originalTextValueMatrix
        else:
            print "Fitting the data imputer"        
            textValuesImputer = Imputer(missing_values='NaN', strategy='median', axis=0)
            self.imputedTextValueMatrix = textValuesImputer.fit_transform(originalTextValueMatrix)
        
        # for each property
        for property, trainRegion2value in trainMatrix.items():
            print "Training for ", property#, " with params l1_ratio ", l1_ratio, " and l1_strength ", l1_strength
            # using the median of the property as back up
            self.property2median[property] = numpy.median(trainRegion2value.values())
            
            #self.property2regressor[property] = ElasticNet(l1_strength, l1_ratio)
            self.property2regressor[property] = LinearRegression()
             
            # first construct the target values
            targetValues = []
            # occasionally we have missing values
            trainingVectors = []
            for region, value in trainRegion2value.items():
                if region in self.region2row:# and property =="/location/statistical_region/population":
                    # if neighborhood, subtract the median
                    if self.neighborhoodModel:
                        targetValues.append(value - self.property2median[property])
                    else:
                        targetValues.append(value)
                    trainingVectors.append(self.imputedTextValueMatrix[self.region2row[region]])
                else:
                    print "No text patterns for region ", region.encode('utf-8'), " skipping it in training"
            # Code to inspect the results
            #if property =="/location/statistical_region/population":
            #    print numpy.shape(trainingVectors)
            #    print numpy.shape(targetValues)
            #print trainingVectors
            #print targetValues
            #print numpy.any(numpy.isnan(trainingVectors)) #False
            #print numpy.any(numpy.isinf(trainingVectors)) #False
            #print numpy.all(numpy.isfinite(trainingVectors)) #True
            self.property2regressor[property].fit(trainingVectors, targetValues)
                # this should print the weights learnt
            #    pattern2weight = {}
            #    for idx, pattern in enumerate(self.column2pattern):
            #        pattern2weight[pattern] = self.property2regressor[property].coef_[idx]
            #    sortedWeights = sorted(pattern2weight.items(), key=operator.itemgetter(1))
            #    for weight in sortedWeights:
            #        print weight
            #    print "intercept:", self.property2regressor[property].intercept_
        
          
if __name__ == "__main__":
    
    import sys

    linearRegressionPredictor = LinearRegressionPredictor()
    
    trainMatrix = linearRegressionPredictor.loadMatrix(sys.argv[1])
    textMatrix = linearRegressionPredictor.loadMatrix(sys.argv[2])
    testMatrix = linearRegressionPredictor.loadMatrix(sys.argv[3])
    
    # experiments with different cut-off thresholds
    bestParams = linearRegressionPredictor.crossValidate(trainMatrix, textMatrix, 4 ,[[2, True],[2, False]])
    linearRegressionPredictor.runEval(trainMatrix, textMatrix, testMatrix, bestParams)