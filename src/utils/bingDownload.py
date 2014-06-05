# Following the example from here
# http://www.cs.columbia.edu/~gravano/cs6111/Proj1/bing-python.txt

import urllib2
import base64

# TODO: these are to be loaded from the jsons
# better lowercase the property names
propertyKeywords = ["population"]
countryNames = ["Czech Republic"]

bingUrl = 'https://api.datamarket.azure.com/Bing/Search/Web' # ?Query=%27gates%27&$top=10&$format=json'
#Provide your account key here
accountKey = 'ZAk6G5VxGSD+K/mx3QH+PX24x85Cx9lEVnQzXA5H+P0'
accountKeyEnc = base64.b64encode(accountKey + ':' + accountKey)
headers = {'Authorization': 'Basic ' + accountKeyEnc}


req = urllib2.Request(bingUrl, headers = headers)
response = urllib2.urlopen(req)
content = response.read()
#content contains the xml/json response from Bing. 
print content