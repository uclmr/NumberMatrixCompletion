NumberMatrixCompletion
======================

Repository for the number matrix completion for freebase data on statistical regions

Preprocessing order:

1. FreeBaseDownload.py: to get the JSONs for all statistical regions in FreeBase.
2. numberExtraction.py: to extract the most recent numbers mentioned for each statistical region in a triple form: region-property-value
3. dataFiltering.py: to get the countries and properties with most values filled in (2 parameters to play with)
4. bingDownload.py: to run queries of the form "region + property" on Bing and get JSONs back with the links
5. htmlDownload.py: to get the html from the links

FreeBaseDownloadMQLonly.py was an attempt to do the FreeBaseDownload using only the MQL API. While it is more elegant to code up, it turns out that the MQL API doesn't have access to all the data in FreeBase, thus there is no point in using it.