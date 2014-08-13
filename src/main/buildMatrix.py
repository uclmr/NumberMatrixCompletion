'''

This script reads in parsed and NER'ed JSONs by Stanford CoreNLP and produces the following structure:

Location:[dep1:[val1, val2], dep1:[val1, val2, ...]]


'''

import json
import sys
import os
import numpy
import glob
import networkx

# this class def allows us to write:
#print(json.dumps(np.arange(5), cls=NumPyArangeEncoder))
#class NumPyArangeEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, numpy.ndarray):
#            return obj.tolist() # or map(int, obj)
#        return json.JSONEncoder.default(self, obj)


parsedJSONDir = sys.argv[1]

# get all the files
jsonFiles = glob.glob(parsedJSONDir + "/*.json")

# one json to rule them all
outputFile = sys.argv[2]

# This is the location2deps2values
theMatrix = {}

print str(len(jsonFiles)) + " files to process"

for jsonFileName in jsonFiles:
    print "processing " + jsonFileName
    with open(jsonFileName) as jsonFile:
        parsedSentences = json.loads(jsonFile.read())
    
    for sentence in parsedSentences:
         
        # get the locations and the numbers
        locationIDs = []
        currentLocation = []
        numberIDs = []
        numbers= []
        for idx, token in enumerate(sentence["tokens"]):
            # if it is a location token add it:
            if token["ner"] == "LOCATION":
                currentLocation.append(idx)
            # if it is a no location token
            else:
                # check if we have just finished a location 
                if len(currentLocation) > 0:
                    locationIDs.append(currentLocation)
                    currentLocation = []
                # we are not in a name so, check for number token:
                try:
                    numbers.append(float(token["word"]))
                    numberIDs.append(idx)
                except ValueError:
                    pass
                                
        #print locationIDs
        #print numberIDs
        
        # if there was at least one location and one number build the dependency graph:
        if len(locationIDs) > 0 and len(numberIDs) > 0:
            # construct the location names
            locations = []
            for locationID in locationIDs:
                locationTokens = []
                for idx in locationID:
                    locationTokens.append(sentence["tokens"][idx]["word"]) 
                locations.append(" ".join(locationTokens))
            #print locations
            #print numbers
            
            sentenceDAG = networkx.DiGraph()
            for idx, token in enumerate(sentence["tokens"]):
                sentenceDAG.add_node(idx, word=token["word"])
        
            # and now the edges:
            for dependency in sentence["dependencies"]:
                sentenceDAG.add_edge(dependency["head"], dependency["dep"], label=dependency["label"])
                
            # add the reverse if one doesn't exist
            for dependency in sentence["dependencies"]:
                if not sentenceDAG.has_edge(dependency["dep"], dependency["head"]):
                    sentenceDAG.add_edge(dependency["dep"], dependency["head"], label="-" + dependency["label"])

            # for each pair of location and number 
            # get the pairs of each and find their dependency paths (might be more than one) 
            for i, locationID in enumerate(locationIDs):
                locationName = locations[i]
                for j, numberID in enumerate(numberIDs):
                    number = numbers[j]
                    # keep all the shortest paths between the number and the tokens of the location
                    shortestPaths = []
                    for tokenID in locationID:
                        try:
                            # get the shortest paths
                            # get the list as it they are unlikely to be very many and we need to len()                  
                            tempShortestPaths = list(networkx.all_shortest_paths(sentenceDAG, source=tokenID, target=numberID))
                            # if the paths found are shorter than the ones we had (or we didn't have any)
                            if (len(shortestPaths) == 0) or len(shortestPaths[0]) > len(tempShortestPaths[0]):
                                shortestPaths = tempShortestPaths
                            # if they have equal length add them
                            elif  len(shortestPaths[0]) == len(tempShortestPaths[0]):
                                shortestPaths.extend(tempShortestPaths)
                        # if not paths were found, do nothing
                        except networkx.exception.NetworkXNoPath:
                            pass
                    for shortestPath in shortestPaths:
                        # get the first dep
                        pathStrings = []
                        pathStrings.append(sentenceDAG[shortestPath[0]][shortestPath[1]]["label"])
                        # for the words in between add the lemma and the dep
                        for seq, tokenIDX in enumerate(shortestPath[1:-1]):
                            pathStrings.append(sentence["tokens"][tokenIDX]["lemma"] + "~" + sentenceDAG[tokenIDX][shortestPath[seq+2]]["label"])
                        pathString = "-".join(pathStrings)
                        #print locationName + ":" + pathString  + ":" + str(number)
                        
                        if pathString not in theMatrix:
                            theMatrix[pathString] = {}
                            
                        if locationName not in theMatrix[pathString]:
                            theMatrix[pathString][locationName] = []
                        
                        theMatrix[pathString][locationName].append(number)
#print theMatrix 
                        
with open(outputFile, "wb") as out:
    json.dump(theMatrix, out)
                        