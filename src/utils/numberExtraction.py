'''
Created on May 10, 2014

@author: andreasvlachos
'''

import json
import os
#import cPickle

# given a json extract the most recent numerical value for each of the properties mentioned
def extractNumericalValues(jsonObj):
    numbers = {}
    country = json.loads(jsonObj)
    if country["name"] == None:
        return None, {}
    name = country["name"].encode('utf-8')
    print name
    # for each property
    # Some countries have nothin in them
    if "property" not in country:
        return name, {}
    for prop, value in country["property"].items():
        # get the name 
        # avoid the religions
        if prop == "/location/statistical_region/religions":
            continue
        print "Property=" + prop
        # keep the valuetype for this property
        #valueType = value["valuetype"]
        #print valueType
        mostRecentValue = 0.0
        # we represent time as year, month
        mostRecentTime = [0, 0]
        for val in value["values"]:
            #print val["property"].keys()
            thisValue = None
            thisTime = None
            if "property" in val:
                for item, v in val["property"].items():
                    #print item
                    if item.split("/")[-1] in ["amount", "rate", "number", "adjusted_value", "cubic_meters", "index_value", "days"]:
                        thisValue = float(v["values"][0]["value"])
                    elif item.split("/")[-1] in ["date", "valid_date", "year", "measurement_date"]:
                        try:
                            thisTime = v["values"][0]["value"]
                        except:
                            #if it fails, i.e. the date has not been properly stated, then do nothing (leave it to None)
                            pass
            
            # if we got a timed value:
            if thisTime != None:
                try:
                    # if the time is given as YYYY-MM or YYYY-MM-DD                             
                    if thisTime.find("-") > -1:
                        if len(thisTime.split("-")) ==2:
                            thisYear, thisMonth = thisTime.split("-")
                            thisTime = [int(thisYear), int(thisMonth)]
                        elif len(thisTime.split("-")) ==3:
                            # the day of the month is ignored
                            thisYear, thisMonth, thisDay = thisTime.split("-")
                            thisTime = [int(thisYear), int(thisMonth)]                        
                    else:
                        # or it is just YYYY
                        thisTime = [int(thisTime), 0]
                    if (mostRecentTime == [0,0]) or (thisTime[0] > mostRecentTime[0]) \
                        or (thisTime[0] == mostRecentTime[0] and thisTime[1] > mostRecentTime[1]):
                        mostRecentTime = thisTime
                        mostRecentValue = thisValue
                # if the time specified cannot be parsed, ignore it
                except ValueError:
                    pass
        # in some cases we do not get any suitable values, r.g. religion
        # or the time of the measurement is too old, say previous decade
        if mostRecentTime != [0,0] and mostRecentTime[0] > 2010:
            print "Time=" + str(mostRecentTime) + " Value=" + str(mostRecentValue)
            numbers[prop] = mostRecentValue
    return name, numbers
    

if __name__ == '__main__':
    dirName = "/cs/research/intelsys/home1/avlachos/FactChecking/FreeBaseData"
    countries2numbers = {}
    totalCountries = 0
    totalNumbers = 0
    uniqueRelations = []
    relation2counts = {}
    rels = []
    for fl in os.listdir(dirName):
        print fl
        jsonFl = open(dirName + "/" + fl).read()
        name, numbers = extractNumericalValues(jsonFl)
        if name != None and len(numbers)>0:
            countries2numbers[name] = numbers
            totalCountries += 1
            for relation in numbers:
                totalNumbers += 1
                if relation not in uniqueRelations:
                    uniqueRelations.append(relation)
                    relation2counts[relation] = 0
                relation2counts[relation] += 1
            
    print countries2numbers
    # maybe return a json? Would be useful to be language independent
    print relation2counts
    print "countries with at least one post 2010 number: " + str(totalCountries)
    print "total post 2010 numbers: " + str(totalNumbers)
    print "Unique relations: " + str(len(uniqueRelations))    
    
    with open(dirName + "/../allCountriesPost2010.json", 'wb') as dc:
        json.dump(countries2numbers, dc)

    
        