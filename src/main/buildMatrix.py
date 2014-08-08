'''

This script reads in parsed and NER'ed JSONs by Stanford CoreNLP and produces the following structure:

Location:[dep1:[val1, val2], dep1:[val1, val2, ...]]


'''

import json
import sys
import os
import numpy

# this class def allows us to write:
#print(json.dumps(np.arange(5), cls=NumPyArangeEncoder))
class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


parsedJSONDir = sys.argv[1]

# get all the files
jsonFiles = glob.glob(parsedJSONDir + "/*")

outputPathName = sys.argv[2]

if not os.path.exists(outputPathName):
    print "creating dir"
    os.mkdir(outputPathName)

# This is the location2deps2values
theMatrix = {}