'''

This script merges the matrix indexed my statistical regions and properties from FreeBase
with the matrix of surface patterns and locations obtained from Stanford CoreNLP

Arg1 is the FreeBase matrix
Arg2 is the Stanford matrix
Arg3 are the aliases

'''

import sys
import json
from networkx.generators.classic import null_graph

# load the file
with open(sys.argv[1]) as freebaseFile:
    region2property2value = json.loads(freebaseFile.read())

# load the file
with open(sys.argv[2]) as jsonFile:
    pattern2location2value = json.loads(jsonFile.read())

# load the file
with open(sys.argv[3]) as jsonFile:
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
for pattern, location2value in pattern2location2value.items():
    # so here are the locations
    locations = location2value.keys()
    # we must be careful in case two or more locations are collapsed to the same region
    region2values = {} 
    # for each location
    for location in locations:
        # if the location has an alias
        if location in alias2region:
            
