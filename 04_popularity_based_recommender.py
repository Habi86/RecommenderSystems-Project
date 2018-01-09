# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
from operator import itemgetter  # for sorting dictionaries w.r.t. values

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM.txt"                # artist-artist similarity matrix (AAM)
METHOD = "PB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]
NF = 2              # number of folds to perform in cross-validation
VERBOSE = False     # verbose output?VERBOSE = False  # verbose output?


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

UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
K = 10 # number of recommendations


def recommend_PB():

    amount_artists = UAM.shape[1]
    UAM_sum = np.sum(UAM, axis=0)
    
    listening_events_per_artist = np.sum(UAM, axis=0)                                    
    recommended_artists_idx = np.argsort(listening_events_per_artist)[-K:]                        
    top_recommendations = np.flipud(recommended_artists_idx)
    sum = UAM_sum[top_recommendations]
    print "top recommendations: "
    print(top_recommendations)

    return len(top_recommendations)

recommend_PB()