# Content-based recommender:
# First, decide on one (or more) data source(s) to acquire external data on music items, e.g., web pages about the artists returned by a search engine, lyrics of the artists, or microblogs about the artists.
# Write a crawler that automatically fetches the respective ('music context') data for all artists in the collection.
# Then, create a representation of the artists inferred from your crawled data: e.g., term weight vectors according to the vector space model or co-occurrence information.
# Depending on your artist representation, choose a suited similarity measure and compute pairwise similarities between the artists:
# e.g., cosine similarity on term weight vectors, co-occurrence likelihood, or set-based Jaccard index.
# Finally, build a content-based recommender using the similarity matrix created.

####################
####################
# WIKIPEDIA FETCHER
####################
####################
# Wikipedia fetcher to download "music context" data for a given artist list.
__author__ = 'Markus Schedl' # modified

# Load required modules
import os
import urllib
import csv
import random

# Parameters
WIKIPEDIA_URL = "http://en.wikipedia.org/wiki/"

ARTISTS_FILE = "./data/LFM1b_artists.txt"                   # text file containing Last.fm user names
OUTPUT_DIRECTORY = "./data/wikipedia/crawls_wikipedia_"     # directory to write output to

USE_INDEX_IN_OUTPUT_FILE = True             # use [index].html as output file name (if set to False, the url-encoded artist name is used)
SKIP_EXISTING_FILES = True                  # skip files already retrieved



# Simple function to read content of a text file into a list
def read_file(fn):
    items = []                                      # list to hold artist names
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
        #reader.next()                              # in case we have a header --> we don't
        for row in reader:
            items.append(row[1])                    # switched from 0 to 1 to only save the name not the idx, cz id + name is saved in txt-file
    return items

# Function to fetch a Wikipedia page, using artist name as query input
def fetch_wikipedia_page(query):
    # retrieve content from URL
    query_quoted = urllib.quote(query)

    url = WIKIPEDIA_URL + query_quoted
    try:
        print "Retrieving data from " + url
        content = urllib.urlopen(url).read()
        return content
    except IOError:                     # return empty content in case some IO / socket error occurred
        return ""


# Main program
if __name__ == '__main__':

    # Create output directory if non-existent
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Read artist list
    artists = read_file(ARTISTS_FILE)
    artistsSample = []

    for i in range(0, 100):
        artistsSample.append(random.choice(artists))

    # Retrieve Wikipedia pages for all artists
    # Either use index as output file name
    if USE_INDEX_IN_OUTPUT_FILE:
        for i in range(0, len(artists)):
            html_fn = OUTPUT_DIRECTORY + "/" + str(i) + ".html"     # target file name
            # check if file already exists
            if os.path.exists(html_fn) & SKIP_EXISTING_FILES:       # if so and it should be skipped, skip the file
                print "File already fetched: " + html_fn
                continue
            # otherwise, fetch HTML content
            html_content = fetch_wikipedia_page(artists[i])
            # write to output file
            print "Storing content to " + html_fn
            with open(html_fn, 'w') as f:
                f.write(html_content)
    else:
        # Or use url-encoded artist name
        for a in artists:
            html_fn = OUTPUT_DIRECTORY + "/" + urllib.quote(a) + ".html"     # target file name
            # check if file already exists
            if os.path.exists(html_fn) & SKIP_EXISTING_FILES:       # if so and it should be skipped, skip the file
                print "File already fetched: " + html_fn
                continue
            # otherwise, fetch HTML content
            html_content = fetch_wikipedia_page(a)
            # write to output file
            print "Storing content to " + html_fn
            with open(html_fn, 'w') as f:
                f.write(html_content)
































