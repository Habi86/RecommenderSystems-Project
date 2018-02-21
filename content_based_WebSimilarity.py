# Post-process the crawled music context data, extract term weights, and compute cosine similarities.

# Load required modules
import os
import numpy as np
import scipy.spatial.distance as scidist
import content_based_Wikipedia_Fetcher as Wikipedia_Fetcher
from lxml.html.clean import Cleaner #https://stackoverflow.com/questions/3073881/clean-up-html-in-python

# Parameters
OUTPUT_TFIDF_FILE = "./data/wikipedia/tfidfs.txt"            # file to store term weights
OUTPUT_TERMS_FILE = "./data/wikipedia/terms.txt"             # file to store list of terms (for easy interpretation of term weights)
OUTPUT_SIMS_FILE = "./data/wikipedia/AAM.txt"                # file to store similarities between items

# Stop words used by Google
STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

def sanitize(dirty_html):
    cleaner = Cleaner(page_structure=True,
                  meta=True,
                  embedded=True,
                  links=True,
                  style=True,
                  processing_instructions=True,
                  inline_style=True,
                  scripts=True,
                  javascript=True,
                  comments=True,
                  frames=True,
                  forms=True,
                  annoying_tags=True,
                  remove_unknown_tags=True,
                  safe_attrs_only=True,
                  #safe_attrs=frozenset(['src','color', 'href', 'title', 'class', 'name', 'id']),
                  remove_tags=('span', 'font', 'div', 'p', 'ul', 'td', 'th', 'tr', 'li', 'a')
                  )
    return cleaner.clean_html(dirty_html)


if __name__ == '__main__':
    html_contents = {}     # dictionary to hold tokenized HTML content of each artist
    terms_df = {}          # dictionary to hold document frequency of each term in corpus
    term_list = []         # list of all terms

    artists = Wikipedia_Fetcher.read_file(Wikipedia_Fetcher.IDX_ARTISTS_FILE)
    artists_len = len(artists)

    for i in range(0, len(artists)):
        html_fn = Wikipedia_Fetcher.OUTPUT_DIRECTORY + "/" + artists[i][0] + ".html"     # target file name

        if os.path.exists(html_fn):
            html_content = open(html_fn, 'r').read()

            # text processing:
            content_tags_removed = sanitize(html_content)                                      # Strip content off HTML tags
            content_casefolded = content_tags_removed.lower()                                  # Perform case-folding, i.e., convert to lower case
            tokens = content_casefolded.split()                                                # Tokenize stripped content at white space characters
            tokens_filtered = filter(lambda t: t.isalnum(), tokens)                            # Remove all tokens containing non-alphanumeric characters; using a simple lambda function (i.e., anonymous function, can be used as parameter to other function)
            tokens_filtered_stopped = filter(lambda t: t not in STOP_WORDS, tokens_filtered)   # Remove words in the stop word list

            html_contents[i] = tokens_filtered_stopped
            print "File " + html_fn + " --- total tokens: " + str(len(tokens)) + "; after filtering and stopping: " + str(len(tokens_filtered_stopped))
        else:
            print "Target file " + html_fn + " does not exist!"


    # Start computing term weights, in particular, document frequencies and term frequencies.
    for aid, terms in html_contents.items():     # Iterate over all (key, value) tuples from dictionary just created to determine document frequency (DF) of all terms
        for t in set(terms):                     # convert list of terms to set of terms ("uniquify" words for each artist/document)
            if t not in terms_df:
                terms_df[t] = 1
            else:
                terms_df[t] += 1


    # Compute number of artists/documents and terms
    #no_artists = len(html_contents.items())
    no_artists = artists_len
    no_terms = len(terms_df)
    print "Number of artists in corpus: " + str(no_artists)
    print "Number of terms in corpus: " + str(no_terms)

    # dimensionality reduction; filtering all terms with a very small document frequency.
    term_list = filter(lambda t: terms_df[t] != 1, terms_df)       # !

    for t in terms_df.keys():     # Dictionary is unordered, so we store all terms in a list to fix their order, before computing the TF-IDF matrix
        term_list.append(t)

    idf = np.zeros(no_terms, dtype=np.float32)     # Create IDF vector using logarithmic IDF formulation
    for i in range(0, no_terms):
        idf[i] = np.log(no_artists / terms_df[term_list[i]])

    tfidf = np.zeros(shape=(no_artists, no_terms), dtype=np.float32)     # Initialize matrix to hold term frequencies (and eventually TF-IDF weights) for all artists for which we fetched HTML content

    # Iterate over all (artist, terms) tuples to determine all term frequencies TF_{artist,term}
    terms_index_lookup = {}         # lookup table for indices (for higher efficiency)
    for a_idx, terms in html_contents.items():

        print "Computing term weights for artist " + str(a_idx)
        for t in terms:
            if t in terms_index_lookup:
                t_idx = terms_index_lookup[t]
            else:
                t_idx = term_list.index(t)
                terms_index_lookup[t] = t_idx
            tfidf[a_idx, t_idx] += 1        # increase TF value for every encounter of a term t within a document of the current artist

    # Replace TF values in tfidf by TF-IDF values:
    # copy and reshape IDF vector and point-wise multiply it with the TF values
    tfidf = np.log1p(tfidf) * np.tile(idf, no_artists).reshape(no_artists, no_terms)

    print "Saving TF-IDF matrix to " + OUTPUT_TFIDF_FILE + "."
    np.savetxt(OUTPUT_TFIDF_FILE, tfidf, fmt='%0.6f', delimiter='\t', newline='\n')

    print "Saving term list to " + OUTPUT_TERMS_FILE + "."
    with open(OUTPUT_TERMS_FILE, 'w') as f:
        for t in term_list:
            f.write(t + "\n")

    # Computing cosine similarities and store them
    # print "Computing cosine similarities between artists."
    sims = np.zeros(shape=(no_artists, no_artists), dtype=np.float32)
    for i in range(0, no_artists):     # Compute pairwise similarities between artists
        print "Computing similarities for artist " + str(i)
        for j in range(i, no_artists):
            cossim = 1.0 - scidist.cosine(tfidf[i], tfidf[j])

            # If either TF-IDF vector (of i or j) only contains zeros, cosine similarity is not defined (NaN: not a number).
            # In this case, similarity between i and j is set to zero (or left at zero, in our case).
            if not np.isnan(cossim):
                sims[i, j] = cossim
                sims[j, i] = cossim

    print "Saving cosine similarities to " + OUTPUT_SIMS_FILE + "."
    np.savetxt(OUTPUT_SIMS_FILE, sims, fmt='%0.6f', delimiter='\t', newline='\n')
