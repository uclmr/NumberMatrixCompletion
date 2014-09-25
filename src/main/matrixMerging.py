'''

This script merges the matrix indexed my statistical regions and properties from FreeBase
with the matrix of surface patterns and locations obtained from Stanford CoreNLP

Arg1 is the FreeBase matrix
Arg2 is the Stanford matrix
Arg3 are the aliases

'''

import sys
import json

# load the file
with open(sys.argv[1]) as freebaseFile:
    region2property2value = json.loads(freebaseFile.read())

# load the file
with open(sys.argv[2]) as jsonFile:
    pattern2locations2values = json.loads(jsonFile.read())

# load the file
with open(sys.argv[3]) as jsonFile:
    location2aliases = json.loads(jsonFile.read())
