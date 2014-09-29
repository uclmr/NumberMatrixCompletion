# so this script takes as input a dictionary in json with the following structure:
# dep or string pattern : {location1:[values], location2:[values]}, etc.
# and does the following kinds of filtering:
# - removes locations that have less than one value for a pattern
# - removes patterns for which location lists are all over the place (high stdev)
# - removes patterns that have fewer than arg1 location

# The second argument is a list of (FreeBase) region names to their aliases which will
# to bring condense the matrix (UK and U.K. becoming the same location), but also they
# prepare us for experiments 


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

# load the file
with open(sys.argv[2]) as jsonFile:
    region2aliases = json.loads(jsonFile.read())

# so we first need to take the location2aliases dict and turn in into aliases to region
alias2region = {} 
for region, aliases in region2aliases.items():
    # add the location as alias to itself
    alias2region[region] = region
    for alias in aliases:
        # so if this alias is used for a different location
        if alias in alias2region and region!=alias2region[alias]:            
            alias2region[alias] = None
            alias2region[alias.lower()] = None
        else:
            # remember to add the lower
            alias2region[alias] = region
            alias2region[alias.lower()] = region
            
# now filter out the Nones
for alias, region in alias2region.items():
    if region == None:
        print "alias ", alias, " ambiguous"
        del alias2region[alias]

# ok, let's traverse now all the patterns and any locations we find we match them case independently to the aliases and replace them with the location
for pattern, locations2values in pattern2locations2values.items():
    # so here are the locations
    # we must be careful in case two or more locations are collapsed to the same region
    regions2values = {} 
    # for each location
    for location, values in locations2values.items():
        # if the location has an alias
        if location in alias2region or location.lower() in alias2region:
            # get it
            region = alias2region[location]
            # if we haven't added it to the regions
            if region not in region2values:
                region2values[region] = values
            else:
                region2values[region] = values + region2values[region]
    # replace the location values of the pattern with the new ones
    pattern2locations2values[pattern] = regions2values


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

with open(sys.argv[3], "wb") as out:
    json.dump(pattern2locations2values, out)
