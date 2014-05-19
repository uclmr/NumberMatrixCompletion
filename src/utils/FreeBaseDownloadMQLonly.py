'''
Created on May 10, 2014

@author: andreasvlachos
'''
import json
import urllib
import time # so that we can use time.sleep(5.0) to be within the limits


propertyTpes2ValueTime = {
                        "/measurement_unit/dated_integer":["number", "year"],\
                        "/measurement_unit/dated_money_value":["amount", "valid_date"],\
                        "/location/co2_emission":["emissions","date"],\
                        "/measurement_unit/dated_float":["number","date"],\
                        "/location/oil_production":["oil_produced","date"],\
                        "/location/natural_gas_production":["natural_gas_produced","date"],\
                        "/location/electricity_production":["electricity_produced","date"],\
                        "/measurement_unit/dated_percentage/":["rate","date"],\
                        "/measurement_unit/recurring_money_value":["amount","date"],\
                        "/measurement_unit/dated_metric_ton":["number","date"],\
                        "/measurement_unit/dated_kgoe":["number","date"], \
                        "/measurement_unit/dated_kilowatt_hour":["number","date"],\
                        "/measurement_unit/dated_money_value":["amount","valid_date"],\
                        "/measurement_unit/adjusted_money_value":["adjusted_value","measurement_date"],\
                        "/measurement_unit/dated_metric_tons_per_million_ppp_dollars":["emission_intensity_value","date"],\
                        "/measurement_unit/dated_cubic_meters/":["cubic_meters", "date"],\
                        "/measurement_unit/dated_days/":["days","date"],\
                        "/measurement_unit/dated_index_value":["index_value","date"]
                        }

apiKey = "AIzaSyClJFx89pJR0_8yc1nvTClMUzFPj0r1dHA"
mqlreadUrl = 'https://www.googleapis.com/freebase/v1/mqlread'


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
    if property["expected_type"] in propertyTpes2ValueTime:
        propertiesOfInterest[property["id"]] = {"name" : property["name"], "expected_type":property["expected_type"]}
    
print propertiesOfInterest
print len(propertiesOfInterest)

# TODO: should check, the count seems a bit low
# TODO: re-write the code below to use mql queries for everything

# use the mid instead of the id as they do need escaping
#mql_query = '[{"mid": null,"name": null, "type": "/location/statistical_region","limit": 100}]'
# set this to the last value we obtained
#cursor = ""


# params = {
#   'key': api_key,
#   'query': json.dumps(query),
# }

# Given the quota, we can run this 1000 times daily.
# It stops when the topics are exhausted.

# for i in xrange(1000):
#     # construct the query
#     mql_url = mqlread_url + '?query=' + mql_query + "&cursor=" + cursor
#     print mql_url
#     statisticalRegionsResult = json.loads(urllib.urlopen(mql_url).read())
#     print statisticalRegionsResult
#     for region in statisticalRegionsResult["result"]:
#         print region["mid"]  # + ":" + region["name"]
#         # now get the statistical properties
#         topic_url = topicService_url + region["mid"] + '?' + urllib.urlencode(params)
#         topicResult = json.loads(urllib.urlopen(topic_url).read())
#         # print topicResult
#         topicResult["name"] = region["name"]
#         filename = region["mid"].split("/")[-1]
#         with open(filename, 'w') as outfile:
#             json.dump(topicResult, outfile)
# 
#     # update the cursor
#     cursor = statisticalRegionsResult['cursor']
#     # this cursor can be used to resume the data download
#     print "New cursor to process"
#     print cursor
#     if not cursor:
#         break
