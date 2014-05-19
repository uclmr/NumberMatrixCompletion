""" This script takes the json extracted from the freebase jsons
 and creates a matrix of countries x FreeBase relations.
 We probably want to filter out relations and countries that do not 
 have a lot of values in the data"""
 
import json
from collections import Counter

with open("/cs/research/intelsys/home1/avlachos/FactChecking/allCountriesPost2010.json") as dataFile:
    data = json.loads(dataFile.read())
    
print json.dumps(data, sort_keys=True, indent=4)
    
featureCounts = Counter()
countryCounts = Counter()
for country, numbers in data.items():
    countryCounts[country] = len(numbers)
    for feature in numbers:
        featureCounts[feature] += 1

print countryCounts
print featureCounts 
print len(featureCounts)
print data["Algeria"]
print data["Germany"]["/location/statistical_region/population"]
print data["Algeria"]["/location/statistical_region/population"]

print featureCounts.most_common(10)
 