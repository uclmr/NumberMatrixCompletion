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

# from here get the property ids
with open("/cs/research/intelsys/home1/avlachos/FactChecking/featuresKept.json") as dataFile:
    featuresKept = json.loads(dataFile.read())

# better lowercase the property names
with open("/cs/research/intelsys/home1/avlachos/FactChecking/allStatisticalRegionProperties.json") as dataFile:
    featuresDesc = json.loads(dataFile.read())



propertyKeywords = []
for feature in featuresKept:
    # get the name for it and lower case it
    for feat in featuresDesc["result"]:
        if feat["id"] == feature:
            propertyKeywords.append(feat["name"].lower().encode('utf-8'))
#print propertyKeywords
#propertyKeywords = ["population"]

countryNames = []
for country in allCountriesData:
    countryNames.append(country.encode('utf-8'))
#print countryNames
#countryNames = ["Czech Republic"]

bingUrl = 'https://api.datamarket.azure.com/Bing/SearchWeb/v1/Web' # ?Query=%27gates%27&$top=10&$format=json'
#Provide your account key here
accountKey = 'ZAk6G5VxGSD+K/mx3QH+PX24x85Cx9lEVnQzXA5H+P0'
accountKeyEnc = base64.b64encode(accountKey + ':' + accountKey)
headers = {'Authorization': 'Basic ' + accountKeyEnc}

pathName = "/cs/research/intelsys/home1/avlachos/FactChecking/Bing"

if not os.path.exists(pathName):
    print "creating dir"
    os.mkdir(pathName)

for name in countryNames:
    print name
    for keywords in propertyKeywords:
        print keywords
        params = {
                 #'format': "Json",
                  'Adult': "\'Strict\'",
                  'WebFileType' : "\'HTML\'"
                  }
        # the query terms are done with urllib quote in order to get %20 instead of + (bing likes that instead)
        # Note that add the current year (2014) since Bing API doesn't allow us search only recent documents
        #print '\''.encode('utf-8') + name + " " + keywords + u' 2014\''.encode('utf-8')
        url = bingUrl + "?Query=" + urllib.quote('\''.encode('utf-8') + name + " " + keywords + ' 2014\''.encode('utf-8')) + "&" + urllib.urlencode(params) + "&$format=json"
        print url
        req = urllib2.Request(url, headers = headers)
        response = urllib2.urlopen(req)
        content = json.loads(response.read())
        # content contains the xml/json response from Bing. 
        print content
        # save the json in a file named after the country and the property.        
        filename = pathName + "/" + name + "_" + keywords + ".json"
        with open(filename, 'w') as outfile:
            json.dump(content["d"]["results"], outfile)
        
