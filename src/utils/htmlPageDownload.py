# -*- coding: utf-8 -*-

# This script collects data using Bing for a relation given a set of examples
# Fetch documents that are likely to contain the answers:
# This part should be identical for both training and testing:


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

import sys
import unicodedata
import codecs

import requests
bingAccountKey = 'wGWXVv4CSBVv2feM/6uelkSJIwMYgPHdCUJNz10CLYA='
maxPagesPerQuery = 6
bingStringPrefix = 'https://api.datamarket.azure.com/Data.ashx/Bing/SearchWeb/v1/Web?Query=%27'
bingStringMidfix = '%27&WebSearchOptions=%27DisableQueryAlterations%27&$top='
bingStringSkip = '&$skip='
bingStringPosFix = '&$format=json'

import urllib2
import os.path
import os

# Files to ignore as they are not HTML
# Maybe add .php, .txt
filetypes_to_ignore = ['.pdf', '.txt', '.php', '.asp', '.stm', '.xls', '.csv', '.doc', '.ppt', '.sql']

example_filename = sys.argv[1]

# The filename is the name of the relation with underscores separating words

relation_tokens = os.path.basename(example_filename).split("_")
# generate the query expansions based on the relation tokens:
query_expansions = [x for x in powerset(relation_tokens)]

print query_expansions

# create a dir for the relation:
example_dir = os.path.dirname(example_filename)
relation_dir = os.path.join(example_dir, os.path.basename(example_filename) + "_dir")
if not os.path.exists(relation_dir):
    os.makedirs(relation_dir)

# Get the examples:
example_lines = codecs.open(example_filename, encoding='utf-8').readlines()

for line in example_lines:
    # split the lines in fields
    entityName = line.strip()
    #replace some special characters for bing:
    bingEntityName = entityName.replace('\'', '%27%27')
    bingEntityName = bingEntityName.replace(u'é', '%c3%a9').decode('utf-8')
    print entityName# + " " + filler
    
    instance_dir = os.path.join(relation_dir, entityName)
    os.makedirs(instance_dir)
    # keep a list to avoid processing the same files again and again
    urls_processed = []
    # for each of the relation tokens (and the empty one) make a query:
    for query_exp in query_expansions:
        print '"' + entityName + '"' + ' Edinburgh ' + ' '.join(query_exp)
        # build the Bing query
        queryString = '%22' + '%20'.join(bingEntityName.split()) + '%22%20Edinburgh%20' + '%20'.join(query_exp)
        print queryString
        # run the query and fetch the results
        pages = 0
        # get the urls
        urlsToProcess = []        
        while True:
            bingQueryString = bingStringPrefix + queryString + bingStringMidfix + '50' + bingStringSkip + str(pages*50) + bingStringPosFix
            print bingQueryString
            results = requests.get(bingQueryString, auth=(bingAccountKey,bingAccountKey))
            for index in xrange(len(results.json['d']['results'])):
                urlsToProcess.append(results.json['d']['results'][index]['Url'])

            pages += 1
            #if pages == maxPagesPerQuery:
            if len(results.json['d']['results']) < 50 or pages == maxPagesPerQuery:
                break

        for url in urlsToProcess:
            url = url.encode('utf-8')
            #print url
            if url not in urls_processed and url[-4:] not in filetypes_to_ignore:
                urls_processed.append(url)
                local_filename = os.path.join(instance_dir, url.replace('/', '|').decode('utf-8'))
                # build a request to fetch the html:
                req = urllib2.Request(url, headers={'User-Agent' : "CityModelPopulator (http://www.cl.cam.ac.uk/~av308/)"})
                try:
                    # fetch the html with time limit 30s
                    page = urllib2.urlopen(req, None, 30)
                    # save it in a UTF8 file
                    local_file = codecs.open(local_filename, encoding='utf-8', mode="w")
                    content = page.read()
                    ucontent = ""
                    try:
                        ucontent = unicode(content, 'utf-8')
                    except UnicodeDecodeError:
                        ucontent = unicode(content, 'latin-1')
                    local_file.write(ucontent)
                    local_file.close()                        

                except:
                    print url + '\tERROR:' + str(sys.exc_info()[0])
