'''
Created on May 10, 2014

@author: andreasvlachos
'''

import json
import os

# given a json extract the most recent numerical value for each of the properties mentioned
def extractNumericalValues(jsonObj):
    country = json.loads(jsonObj)
    name = country["name"]
    print name
    # for each property
    for prop, value in country["property"].items():
        # get the name 
        print "Property=" + prop
        # keep the valuetype for this property
        #valueType = value["valuetype"]
        #print valueType
        mostRecentValue = 0
        mostRecentTime = 0
        for val in value["values"]:
            #print val["property"].keys()
            thisValue = None
            thisTime = None
            for item, v in val["property"].items():
                #print item
                if item.split("/")[-1] in ["amount", "rate", "number", "adjusted_value", "cubic_meters", "index_value", "days"]:
                    thisValue = v["values"][0]["value"]
                elif item.split("/")[-1] in ["date", "valid_date", "year", "measurement_date"]:
                    thisTime = v["values"][0]["value"]
            if thisTime > mostRecentTime:
                mostRecentTime = thisTime
                mostRecentValue = thisValue
        # in some cases we do not get any suitable values, r.g. religion
        if mostRecentTime >0:
            print "Time=" + str(mostRecentTime) + " Value=" + str(mostRecentValue)
                
                
        #TODO: what about things that have months and years, e.g. major_exports, minimum wage    
    

if __name__ == '__main__':
    dirName = "../../../FreeBaseData"
    for fl in os.listdir(dirName):
        jsonFl = open(dirName + "/" + fl).read()
        extractNumericalValues(jsonFl)