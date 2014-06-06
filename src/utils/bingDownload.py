# Following the example from here
# http://www.cs.columbia.edu/~gravano/cs6111/Proj1/bing-python.txt

import urllib2
import base64
import urllib
import json
import os



# from here we only want to keep the countries
with open("/cs/research/intelsys/home1/avlachos/FactChecking/allCountriesPost2010Filtered15-150.json") as dataFile:
    allCountriesData = json.loads(dataFile.read())


with open("/cs/research/intelsys/home1/avlachos/FactChecking/featuresKept.json") as dataFile:
    featuresKept = json.loads(dataFile.read())

# better lowercase the property names
with open("/cs/research/intelsys/home1/avlachos/FactChecking/allStatisticalRegionProperties.json") as dataFile:
    featuresDesc = json.loads(dataFile.read())


#TODO: try this out with a few first, say 3 countries and 3 features to make sure that the pipeline works end to end
propertyKeywords = ["population"]
countryNames = ["Czech Republic"]



bingUrl = 'https://api.datamarket.azure.com/Bing/SearchWeb/v1/Web' # ?Query=%27gates%27&$top=10&$format=json'
#Provide your account key here
accountKey = 'ZAk6G5VxGSD+K/mx3QH+PX24x85Cx9lEVnQzXA5H+P0'
accountKeyEnc = base64.b64encode(accountKey + ':' + accountKey)
headers = {'Authorization': 'Basic ' + accountKeyEnc}

pathName = "/cs/research/intelsys/home1/avlachos/FactChecking/Bing"

if os.path.exists(pathName):
    os.mkdir(pathName)

for name in countryNames:
    for keywords in propertyKeywords:
        params = {
                 #'format': "Json",
                  'Adult': "\'Strict\'",
                  'WebFileType' : "\'HTML\'"
                  }
        # the query terms are done with urllib quote in order to get %20 instead of + (bing likes that instead)
        url = bingUrl + "?Query=" + urllib.quote("\'" + name + " " + keywords + "\'") + "&" + urllib.urlencode(params) + "&$format=json"
        print url
        req = urllib2.Request(url, headers = headers)
        response = urllib2.urlopen(req)
        content = json.loads(response.read())
        #content contains the xml/json response from Bing. 
        print content
        filename = pathName + 
        with open(filename, 'w') as outfile:
            json.dump(content, outfile)
        
        # save the json in a file named after the relation and the property.