# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random


# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        reader.next()  # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data

# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB_artist(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx

# Function that implements a dumb random recommender. It predicts a number of artists from randomly chosen users.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB_user(UAM, seed_aidx_train, no_items, K_users = 1):
    # UAM                   user-artist-matrix
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict
    # K_users               no of random users selected

    # Select a random sample of users
    random_uidx = random.sample(range(0,UAM.shape[0]), K_users)
    # Get artits of these
    random_aidx_nz = np.nonzero(UAM[random_uidx,:])[1]      # only interested in artists, hence [1]
    # Remove artists in training set of seed user
    random_aidx = np.setdiff1d(set(random_aidx_nz), seed_aidx_train)

    if VERBOSE:
        print str(K_users) + ' user(s) randomly chosen, ' + str(no_items) + ' recommendations requested, ' + str(len(random_aidx)) + ' found' # restart with K=' + str(K_users+1)

    # Start over with increased number of users to consider, if recommended artists smaller than requested
    if len(random_aidx) < no_items:
        K_users += 1
        return recommend_RB_user(UAM, seed_aidx_train, no_items, K_users)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx


