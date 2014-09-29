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

# load the FreeBase file
with open(sys.argv[1]) as freebaseFile:
    region2property2value = json.loads(freebaseFile.read())
    
# we need to make it property2region2value
property2region2value = {}
for region, property2value in region2property2value.items():
    for property, value in property2value.items():
        if property not in property2region2value:
            property2region2value[property] = {}
        property2region2value[property][region] = value

# load the file with the surface patterns
with open(sys.argv[2]) as jsonFile:
    pattern2region2value = json.loads(jsonFile.read())

# TODO:
# for each of the FB properties, find the 10 best correlated surface patterns
 
