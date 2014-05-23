'''
Created on May 10, 2014

@author: andreasvlachos

Takes one argument, the output directory
WARNING: The MQL API does not return all the data held in FreeBase, only a substantial (seemingly random) part of it.

'''
import json
import urllib
import time # so that we can use time.sleep(5.0) to be within the limits
import copy
import sys
import os


propertyTypes2ValueTime = {
                        "/measurement_unit/dated_integer":["number", "year"],\
                        "/measurement_unit/dated_money_value":["amount", "valid_date"],\
                        "/location/co2_emission":["emissions","date"],\
                        "/measurement_unit/dated_float":["number","date"],\
                        "/location/oil_production":["oil_produced","date"],\
                        "/location/natural_gas_production":["natural_gas_produced","date"],\
                        "/location/electricity_production":["electricity_produced","date"],\
                        "/measurement_unit/dated_percentage":["rate","date"],\
                        "/measurement_unit/recurring_money_value":["amount","date"],\
                        "/measurement_unit/dated_metric_ton":["number","date"],\
                        "/measurement_unit/dated_kgoe":["number","date"], \
                        "/measurement_unit/dated_kilowatt_hour":["number","date"],\
                        "/measurement_unit/dated_money_value":["amount","valid_date"],\
                        "/measurement_unit/adjusted_money_value":["adjusted_value","measurement_date"],\
                        "/measurement_unit/dated_metric_tons_per_million_ppp_dollars":["emission_intensity_value","date"],\
                        "/measurement_unit/dated_cubic_meters":["cubic_meters", "date"],\
                        "/measurement_unit/dated_days":["days","date"],\
                        "/measurement_unit/dated_index_value":["index_value","date"]
                        }

apiKey = "AIzaSyClJFx89pJR0_8yc1nvTClMUzFPj0r1dHA"
mqlreadUrl = 'https://www.googleapis.com/freebase/v1/mqlread'

outputDir = sys.argv[1]
os.mkdir(outputDir)

# We have three kinds of queries

# First the properties of statistical regions
propertiesQuery = [{ "type": "/type/property", "schema": { "id": "/location/statistical_region" },\
                     "id": None, "name": None, "expected_type": None }]

propertiesQueryParams = {
  'key': apiKey,
  'query': json.dumps(propertiesQuery),
  'limit': 300
}

propertiesUrl = mqlreadUrl + '?' + urllib.urlencode(propertiesQueryParams)

propertiesJSON =  json.loads(urllib.urlopen(propertiesUrl).read())

# Keep here the properties we care for
propertiesOfInterest = {}
for property in propertiesJSON["result"]:
    # if we are handling the expected type of the property
    if property["expected_type"] in propertyTypes2ValueTime:
        propertiesOfInterest[property["id"]] = {"name" : property["name"], "expectedType":property["expected_type"]}
        print property["id"] + " accepted"
    else:
        print property["id"] + " rejected"

# save them in a file
filename = outputDir + "/propertiesOfInterest.json"
with open(filename, 'w') as outfile:
    json.dump(propertiesOfInterest, outfile)


print str(len(propertiesOfInterest)) + " properties accepted"
 

# this is the query to iterate through all statistical regions
statisticalRegionsQuery = [{"mid": None,"name": None, "type": "/location/statistical_region"}]
# set this to the last value we obtained
statisticalRegionsCursor = ""

statisticalRegionsQueryParams = {
   'key': apiKey,
   'query': json.dumps(statisticalRegionsQuery),
   'limit': 100,
}

# we also need to construct the query for all the properties of interest.
# HACK: the query for all propoerties of interest is too long for GET, so broke it down into two parts
dataQueryTemplate1 = [{}] 

for property, features in propertiesOfInterest.items()[:40]:
    expectedType = features["expectedType"]
    # need to add the optional true because some of the properties will not be there
    propertySubQ = {propertyTypes2ValueTime[expectedType][0] : None, propertyTypes2ValueTime[expectedType][1] : None , "sort" : propertyTypes2ValueTime[expectedType][1],"optional": True}
    dataQueryTemplate1[0][property] = [propertySubQ] 

dataQueryTemplate2 = [{}] 

for property, features in propertiesOfInterest.items()[40:]:
    expectedType = features["expectedType"]
    # need to add the optional true because some of the properties will not be there
    propertySubQ = {propertyTypes2ValueTime[expectedType][0] : None, propertyTypes2ValueTime[expectedType][1] : None , "sort" : propertyTypes2ValueTime[expectedType][1],"optional": True}
    dataQueryTemplate2[0][property] = [propertySubQ] 

dataQueryTemplates = [dataQueryTemplate1, dataQueryTemplate2]
       
#print dataQueryTemplate

# we add the sleep so that it runs at about 1000 times daily
# It stops when the topics are exhausted.

while(True):
    startTime = time.time()
    # construct the query
    statisticalRegionsUrl = mqlreadUrl + '?' + urllib.urlencode(statisticalRegionsQueryParams) + "&cursor=" + statisticalRegionsCursor
    #print statisticalRegionsUrl
    statisticalRegionsResult = json.loads(urllib.urlopen(statisticalRegionsUrl).read())
    #print statisticalRegionsResult
    for region in statisticalRegionsResult["result"]:
        print region["mid"] + ":" + region["name"]

        # this will hold the data for the region 
        data = {}
        # keep only the properties that had values
        data["mid"]  = region["mid"]
        data["name"] = region["name"]
        
        for dataQueryTemplate in dataQueryTemplates:
            dataQuery = copy.deepcopy(dataQueryTemplate)
            dataQuery[0]["mid"] = region["mid"]
            #print dataQuery
        
            dataQueryParams = {'key': apiKey, 'query': json.dumps(dataQuery)}

            dataQueryUrl = mqlreadUrl + '?' + urllib.urlencode(dataQueryParams)
            #print dataQueryUrl
            #print urllib.urlopen(dataQueryUrl).read()
            dataResult = json.loads(urllib.urlopen(dataQueryUrl).read())
        
            # this might do with POST but it does not work...:
            #dataRequestObject = urllib2.Request(mqlreadUrl, urllib.urlencode(dataQueryParams))        
            #dataResult = json.loads(urllib2.urlopen(dataRequestObject).read())
        
            for property, values in dataResult["result"][0].items():
                if len(values) > 0:
                    data[property] = values
        
        filename = outputDir + "/" + region["mid"].split("/")[-1]
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
 
    # update the cursor
    statisticalRegionsCursor = statisticalRegionsResult['cursor']
    # this cursor can be used to resume the data download
    print "New cursor to process"
    print statisticalRegionsCursor
    endTime = time.time()
    print "Seconds for a batch of 100" + str(endTime - startTime) 
    if not statisticalRegionsCursor:
        break
    # (24*60*60) seconds in a day, divided by 1000 such batches we can run per day means we need 86s between batches.
    # Let's see how long each batch takes
    #time.sleep(5)