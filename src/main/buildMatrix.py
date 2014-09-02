'''

This script reads in parsed and NER'ed JSONs by Stanford CoreNLP and produces the following structure:

Location:[dep1:[val1, val2], dep1:[val1, val2, ...]]


'''

import json
import sys
import os
import glob
import networkx
import re

# this class def allows us to write:
#print(json.dumps(np.arange(5), cls=NumPyArangeEncoder))
#class NumPyArangeEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, numpy.ndarray):
#            return obj.tolist() # or map(int, obj)
#        return json.JSONEncoder.default(self, obj)



def getNumbers(sentence):
    tokenID2number = {}
    for idx, token in enumerate(sentence["tokens"]):
        # avoid only tokens known to be dates or part of locations
        # This only takes actual numbers into account thus it ignores things like "one million"
        # and also treating "500 millions" as "500" 
        if token["ner"] not in ["DATE", "LOCATION", "PERSON", "ORGANIZATION", "MISC"]:
            try:
                # this makes sure that 123,123,123.23 which fails the float test, becomes 123123123.23 which is good
                tokenWithoutCommas = float(re.sub(",([0-9][0-9][0-9])", "\g<1>", token["word"]))
                tokenID2number[idx] = float(tokenWithoutCommas) 
            except ValueError:
                pass
    return tokenID2number

def getLocations(sentence):
    # note that a location can span multiple tokens
    tokenIDs2location = {}
    currentLocation = []
    for idx, token in enumerate(sentence["tokens"]):
        # if it is a location token add it:
        if token["ner"] == "LOCATION":
            currentLocation.append(idx)
        # if it is a no location token
        else:
            # check if we have just finished a location             
            if len(currentLocation) > 0:
                # convert the tokenID to a tuple (immutable) and put the name there
                locationTokens = []
                for locIdx in currentLocation:
                    locationTokens.append(sentence["tokens"][locIdx]["word"]) 

                tokenIDs2location[tuple(currentLocation)] = " ".join(locationTokens)
                currentLocation = []
                
    return tokenIDs2location

def buildDAGfromSentence(sentence):
    sentenceDAG = networkx.DiGraph()
    for idx, token in enumerate(sentence["tokens"]):
        sentenceDAG.add_node(idx, word=token["word"])
        sentenceDAG.add_node(idx, lemma=token["lemma"])
        
    # and now the edges:
    for dependency in sentence["dependencies"]:
        sentenceDAG.add_edge(dependency["head"], dependency["dep"], label=dependency["label"])
        # add the reverse if one doesn't exist
        # if an edge exists, the label gets updated, thus the standard edges do 
        if not sentenceDAG.has_edge(dependency["dep"], dependency["head"]):
            sentenceDAG.add_edge(dependency["dep"], dependency["head"], label="-" + dependency["label"])
    return sentenceDAG
            
# getDepPaths
# also there can be more than one paths
def getShortestDepPaths(sentenceDAG, locationTokenIDs, numberTokenID):
    shortestPaths = []
    for locationTokenID in locationTokenIDs:
        try:
            # get the shortest paths
            # get the list as it they are unlikely to be very many and we need to len()                  
            tempShortestPaths = list(networkx.all_shortest_paths(sentenceDAG, source=locationTokenID, target=numberTokenID))
            # if the paths found are shorter than the ones we had (or we didn't have any)
            if (len(shortestPaths) == 0) or len(shortestPaths[0]) > len(tempShortestPaths[0]):
                shortestPaths = tempShortestPaths
            # if they have equal length add them
            elif  len(shortestPaths[0]) == len(tempShortestPaths[0]):
                shortestPaths.extend(tempShortestPaths)
        # if not paths were found, do nothing
        except networkx.exception.NetworkXNoPath:
            pass
    return shortestPaths

# given the a dep path defined by the nodes, get the string of the lexicalized dep path, possibly extended by one more dep
def depPath2StringExtend(sentenceDAG, path, extend=True):
    strings = []
    # this keeps the various bits of the string
    pathStrings = []
    # get the first dep which is from the location
    pathStrings.append("LOCATION~" + sentenceDAG[path[0]][path[1]]["label"])
    # for the words in between add the lemma and the dep
    for seqOnPath, tokenId in enumerate(path[1:-1]):
        # the +2 is because we are already on the second node in the path
        pathStrings.append(sentenceDAG.node[tokenId]["lemma"] + "~" + sentenceDAG[tokenId][path[seqOnPath+2]]["label"])
    # add the number bit
    strings.append("+".join(pathStrings + ["NUMBER"]))
                        
    if extend:                            
        # create additional paths by adding all out-edges from the number token (except for the one taking as back)
        # the number token is the last one on the path
        outEdgesFromNumber = sentenceDAG.out_edges_iter([path[-1]])
        for edge in outEdgesFromNumber:
            # the source of the edge we knew
            dummy, outNode = edge
            # if we are not going back:
            if outNode != path[-2]:
                strings.append("+".join(pathStrings + ["NUMBER~" + sentenceDAG[path[-1]][outNode]["label"] + "~" + sentenceDAG.node[outNode]["lemma"] ]))

    return strings

def getSurfacePatternsExtend(sentence, locationTokenIDs, numberTokenID, extend=True):
    # so this can go either from the location to the number, or the other way around
    # if the number token is before the first token of the location
    tokenSeqs = []
    if numberTokenID < locationTokenIDs[0]:
        tokenIDs = range(numberTokenID+1, locationTokenIDs[0])
    else:
        tokenIDs = range(locationTokenIDs[-1]+1, numberTokenID)
    
    tokens = []
    for id in tokenIDs:
        tokens.append('"' + sentence["tokens"][id]["word"] + '"')
     
    if numberTokenID < locationTokenIDs[0]:
        tokens = ["NUMBER"] + tokens + ["LOCATION"]
    else:
        tokens = ["LOCATION"] + tokens + ["NUMBER"]
    tokenSeqs.append(tokens)
    
    if extend:
        lhsID = min([numberTokenID] + list(locationTokenIDs))
        rhsID = max([numberTokenID] + list(locationTokenIDs))
        if lhsID > 1:
            tokenSeqs.append(['"' + sentence["tokens"][lhsID-2]["word"] + '"', '"' + sentence["tokens"][lhsID-1]["word"] + '"'] + tokens)
        elif lhsID == 1:
            tokenSeqs.append(['"' + sentence["tokens"][lhsID-1]["word"] + '"'] + tokens)

        if rhsID < len(sentence["tokens"]) - 2:
            tokenSeqs.append(tokens + ['"' + sentence["tokens"][rhsID+1]["word"] + '"', '"' + sentence["tokens"][rhsID+2]["word"] + '"'])
        elif rhsID == len(sentence["tokens"]) - 2:
            tokenSeqs.append(tokens + ['"' + sentence["tokens"][rhsID+1]["word"] + '"'])
    return tokenSeqs
    
    

# again, we want to extend them on either side.

parsedJSONDir = sys.argv[1]

# get all the files
jsonFiles = glob.glob(parsedJSONDir + "/*.json")

# one json to rule them all
outputFile = sys.argv[2]

# this forms the columns using the lexicalized dependency paths
depPath2location2values = {}
# this forms the columns using the surface patterns
string2location2values = {}

print str(len(jsonFiles)) + " files to process"

for jsonFileName in jsonFiles:
    print "processing " + jsonFileName
    with open(jsonFileName) as jsonFile:
        parsedSentences = json.loads(jsonFile.read())
    
    for sentence in parsedSentences:
        
        tokenID2number = getNumbers(sentence)
        tokenIDs2location = getLocations(sentence)
        
        # if there was at least one location and one number build the dependency graph:
        if len(tokenID2number) > 0 and len(tokenIDs2location) > 0:
            
            sentenceDAG = buildDAGfromSentence(sentence)

            # for each pair of location and number 
            # get the pairs of each and find their dependency paths (might be more than one) 
            for locationTokenIDs, location in tokenIDs2location.items():

                for numberTokenID, number in tokenID2number.items():

                    # keep all the shortest paths between the number and the tokens of the location
                    shortestPaths = getShortestDepPaths(sentenceDAG,  locationTokenIDs, numberTokenID)
                    
                    # ignore paths longer than some number deps (=tokens_on_path + 1)
                    if len(shortestPaths) > 0 and len(shortestPaths[0]) < 10:
                        for shortestPath in shortestPaths:
                            pathStrings = depPath2StringExtend(sentenceDAG, shortestPath)
                            for pathString in pathStrings:
                                if pathString not in depPath2location2values:
                                    depPath2location2values[pathString] = {}
                            
                                if location not in depPath2location2values[pathString]:
                                    depPath2location2values[pathString][location] = []
                        
                                depPath2location2values[pathString][location].append(number)
                                
                    # now get the surface strings 
                    surfacePatternTokenSeqs = getSurfacePatternsExtend(sentence, locationTokenIDs, numberTokenID)   
                    for surfacePatternTokens in surfacePatternTokenSeqs:
                        if len(surfacePatternTokens) < 15:
                            surfaceString = ",".join(surfacePatternTokens)
                            if surfaceString not in string2location2values:
                                string2location2values[surfaceString] = {}
                            
                            if location not in string2location2values[surfaceString]:
                                string2location2values[surfaceString][location] = []
                        
                            string2location2values[surfaceString][location].append(number)
                        

                        
with open(outputFile + "_deps.json", "wb") as out:
    json.dump(depPath2location2values, out)

with open(outputFile + "_strs.json", "wb") as out:
    json.dump(string2location2values, out)

    
# print the deps with the most locations:
dep2counts = {}
for dep, values in depPath2location2values.items():
    dep2counts[dep] = len(values)
    
import operator
sortedDeps = sorted(dep2counts.iteritems(), key=operator.itemgetter(1), reverse=True)

for dep in sortedDeps[:50]:
    print dep[0]
    print depPath2location2values[dep[0]]
    
# print the string with the most locations:
str2counts = {}
for str, values in string2location2values.items():
    str2counts[str] = len(values)
    
sortedStrs = sorted(str2counts.iteritems(), key=operator.itemgetter(1), reverse=True)

for str in sortedStrs[:50]:
    print str[0]
    print string2location2values[str[0]]