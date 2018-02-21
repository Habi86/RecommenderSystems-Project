# Content-based recommender:
# First, decide on one (or more) data source(s) to acquire external data on music items, e.g., web pages about the artists returned by a search engine, lyrics of the artists, or microblogs about the artists.
# Write a crawler that automatically fetches the respective ('music context') data for all artists in the collection.
# Then, create a representation of the artists inferred from your crawled data: e.g., term weight vectors according to the vector space model or co-occurrence information.
# Depending on your artist representation, choose a suited similarity measure and compute pairwise similarities between the artists:
# e.g., cosine similarity on term weight vectors, co-occurrence likelihood, or set-based Jaccard index.
# Finally, build a content-based recommender using the similarity matrix created.

# Wikipedia fetcher to download "music context" data for a given artist list.

# Load required modules
import os
import urllib
import csv

# Parameters
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/"
IDX_ARTISTS_FILE = "./data/C1ku_idx_artists.txt"
ARTISTS_FILE = "./data/LFM1b_artists.txt"                           # text file containing Last.fm user names
ARTISTS_UAMONLY = "./data/artists_names_UAMonly.txt"
OUTPUT_DIRECTORY = "./data/wikipedia/crawls_wikipedia_UAMONLY"      # directory to write output to

USE_INDEX_IN_OUTPUT_FILE = True                                     # use [index].html as output file name (if set to False, the url-encoded artist name is used)
SKIP_EXISTING_FILES = True                                          # skip files already retrieved


def read_file(fn):
    items = []
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            items.append(row)
    return items

def fetch_wikipedia_page(query):
    query_quoted = urllib.quote(query)
    url = WIKIPEDIA_URL + query_quoted
    try:
        content = urllib.urlopen(url).read()
        return content
    except IOError:
        return ""


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIRECTORY):  # Create output directory if non-existent
        os.makedirs(OUTPUT_DIRECTORY)

    artists = read_file(ARTISTS_UAMONLY)

    for i in range(0, len(artists)):
        html_fn = OUTPUT_DIRECTORY + "/" + artists[i][0] + ".html"

        if os.path.exists(html_fn) & SKIP_EXISTING_FILES:
            continue
        html_content = fetch_wikipedia_page(artists[i][1])

        if "Wikipedia does not have an article with this exact name" not in html_content:
            if "may refer to" not in html_content:
                if "Genres" in html_content:
                    with open(html_fn, 'w') as f:
                        f.write(html_content)
