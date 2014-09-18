# so this script takes as input a dictionary in json with the following structure:
# dep or string pattern : {location1:[values], location2:[values]}, etc.
# and does the following kinds of filtering:
# - removes locations that have less than one value for a pattern
# - removes patterns for which location lists are all over the place (high stdev)
# - removes patterns that have fewer than arg1 location


import json
import numpy
import sys

minNumberOfValues = 2
maxAllowedDeviation = 0.05
minNumberOfLocations = 2


# load the file
with open(sys.argv[1]) as jsonFile:
    pattern2locations2values = json.loads(jsonFile.read())

print "patterns before filtering:", len(pattern2locations2values)

countNotEnoughValues = 0
for pattern, loc2values in pattern2locations2values.items():
    for loc in loc2values.keys():
        # so if there are not enough values, remove the location from that pattern
        if len(loc2values[loc]) < minNumberOfValues:
            del loc2values[loc]
            countNotEnoughValues +=1
            
print "set of values removed for having less than", minNumberOfValues, " of values: ", countNotEnoughValues
            
countTooMuchDeviation = 0 
for pattern, loc2values in pattern2locations2values.items():
    for loc, values in loc2values.items():
        a = numpy.array(values)
        # if the values have a high stdev after normalizing them between 0 and 1 (only positive values)
        # the value should be interpreted as the percentage of the max value allowed as stdev
        if numpy.std(a/a.max()) > maxAllowedDeviation:
            del loc2values[loc]
            countTooMuchDeviation += 1
            
print "sets of values removed for having more than", maxAllowedDeviation, " std deviation : ", countTooMuchDeviation            

    
for pattern in pattern2locations2values.keys():
    # now make sure there are enough locations left per pattern
    if len(pattern2locations2values[pattern]) < minNumberOfLocations:
        del pattern2locations2values[pattern]
    else:
        # if there are enough values then just keep the average
        for location in pattern2locations2values[pattern].keys():
            pattern2locations2values[pattern][location] = numpy.mean(pattern2locations2values[pattern][location])
        
print "patterns after filtering:",len(pattern2locations2values)

with open(sys.argv[2], "wb") as out:
    json.dump(pattern2locations2values, out)
