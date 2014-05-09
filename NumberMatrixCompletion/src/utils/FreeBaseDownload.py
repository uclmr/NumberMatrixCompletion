'''
Created on May 10, 2014

@author: andreasvlachos
'''
import json
import urllib

api_key = "AIzaSyClJFx89pJR0_8yc1nvTClMUzFPj0r1dHA"
mqlread_url = 'https://www.googleapis.com/freebase/v1/mqlread'
# use the mid instead of the id as they do need escaping
mql_query = '[{"mid": null,"name": null, "type": "/location/statistical_region","limit": 100}]'
cursor = "eNpVjEEKwkAQBL-jyJKeyc7uzCDiP0IOcWUhICYkxKNvV3IQ7GMV1WVb1mlxw616NzmrRaDvnk7QqCTE0o_r4Ag7QE5k_jiHw2uc3UyFkuSfY_hyMgYPWvA9QopUFX8r4bJ3MbbHhkTabAL2d-jW2dkSEvXNNakpRdGKfK8FliPSB9DJKhs="

api_key = "AIzaSyClJFx89pJR0_8yc1nvTClMUzFPj0r1dHA"
topicService_url = 'https://www.googleapis.com/freebase/v1/topic'
params = {
  'key': api_key,
  'filter': '/location/statistical_region'
}

# Given the quota, we can run this 1000 times daily.
# It stops when the topics are exhausted.

for i in xrange(1000):
    # construct the query
    mql_url = mqlread_url + '?query=' + mql_query + "&cursor=" + cursor
    print mql_url
    statisticalRegionsResult = json.loads(urllib.urlopen(mql_url).read())
    print statisticalRegionsResult
    for region in statisticalRegionsResult["result"]:
        print region["mid"]# + ":" + region["name"]
        # now get the statistical properties
        topic_url = topicService_url +  region["mid"] + '?' + urllib.urlencode(params)
        topicResult = json.loads(urllib.urlopen(topic_url).read())
        #print topicResult
        topicResult["name"] = region["name"]
        filename = region["mid"].split("/")[-1]
        with open(filename, 'w') as outfile:
            json.dump(topicResult, outfile)

    # update the cursor
    cursor = statisticalRegionsResult['cursor']
    # this gues can be used to resume the data download
    print "New cursor to process"
    print cursor
    if not cursor:
        break
