'''

This script merges the matrix indexed my statistical regions and properties from FreeBase
with the matrix of surface patterns and locations obtained from Stanford CoreNLP

Arg1 is the FreeBase matrix
Arg2 is the Stanford matrix
Arg3 are the aliases

'''

import sys
import json
#from scipy.stats.stats import pearsonr
import heapq
import numpy

# load the FreeBase file
with open(sys.argv[1]) as freebaseFile:
    property2region2value = json.loads(freebaseFile.read())

# load the file with the surface patterns
with open(sys.argv[2]) as jsonFile:
    pattern2region2value = json.loads(jsonFile.read())

for property, region2value in property2region2value.items():
    print property
    # now for pattern we have to construct an array with the values for the same regions
    pattern2correlation = {}
    pattern2meanScaledAbsDiff = {}
    for pattern, patternRegion2value in pattern2region2value.items():
        patternValues = []
        propertyValues = []
        for region, value in region2value.items():
            if region in patternRegion2value:
                patternValues.append(patternRegion2value[region])
                propertyValues.append(value)
        # get the correlation only if there were at least 3
        if len(patternValues) >= 3:
            #print pattern
            #print patternValues
            #print propertyValue
            #corr = pearsonr(patternValues, propertyValues)[0]
            #if not numpy.isnan(corr):
            #    pattern2correlation[pattern] = pearsonr(patternValues, propertyValues)[0]
            scaledAbsDiff = 0.0
            nonZeros = 0
            for idx, val in enumerate(propertyValues):
                if val != 0.0:
                    nonZeros +=1
                    absDiff = abs(val-patternValues[idx])
                    scaledAbsDiff += absDiff/abs(val)
            if nonZeros>0:
                pattern2meanScaledAbsDiff[pattern] = scaledAbsDiff/nonZeros
            #print pattern2correlation[pattern]
    # get the patterns with top 10 correlation with the property
    top10 =  heapq.nsmallest(20, pattern2meanScaledAbsDiff, key=pattern2meanScaledAbsDiff.get)
    #print property
    print region2value
    for pattern in top10:
        print pattern.encode('utf-8')
        print pattern2meanScaledAbsDiff[pattern]
        print pattern2region2value[pattern]
    
        
    
