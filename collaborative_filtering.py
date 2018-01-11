# Implement a user-based, memory-based collaborative filtering artist recommender that supports k- nearest
# neighbors prediction. In addition to the version we implemented in the lab, elaborate and report a
# method to combine the predictions for the same artists among the set of nearest neighbors (e.g.,
# how to deal with an artist that is recommended 10 times by 20 nearest neighbors vs. an artist that is
# recommended only once, but by a neighbor with a music taste very similar to the target user; think of
# combining user similarity and artist frequency). Implement your recommender in a function recommend_CF.
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values
from collections import defaultdict




# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM.txt"                # artist-artist similarity matrix (AAM)
METHOD = "PB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]
NF = 10              # number of folds to perform in cross-validation


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






def recommend_CF(user, copy_UAM, train, K):
    
    user_playcount = copy_UAM[user, :]
    # Remove information on test artists from user
    # user_playcount = np.nonzero(user_playcount)[0]
    test_artists = np.setdiff1d(user_playcount, train)
    test_artists = map(int, test_artists)

    copy_UAM[user, test_artists] = 0.0

   
    users = np.zeros(shape=(copy_UAM.shape[0]), dtype=np.float32)
    # users playcounts normalisieren
    copy_UAM[user,:] = copy_UAM[user,:] / np.sum(copy_UAM[user,:])

    # Compute similarities as inverse cosine distance between user_playcount of user and all users via UAM (assuming that UAM is normalized)
    for u in range(0, copy_UAM.shape[0]):
        users[u] = 1.0 - scidist.cosine(user_playcount, copy_UAM[u,:])
        # print "user u: "
        # print users[u]

    # similarity der user absteigend sortieren
    sort_idx = np.argsort(users) 

    
    recommended_artist = multiple_neighbours(sort_idx, K, copy_UAM)
    simple_neighbour(sort_idx, copy_UAM)

    return recommended_artist



def multiple_neighbours(sort_idx, K, copy_UAM):
    neighbor_idx = sort_idx[-1-K:-1]

    artists_of_neighbours = copy_UAM[neighbor_idx, :]

    artist_dictionary = defaultdict(list)
    for user in artists_of_neighbours:
        for key, value in enumerate(user):
            if(value == 0): continue
            if(artist_dictionary[key]):
                artist_dictionary[key] = artist_dictionary[key] + value
            else:
                artist_dictionary[key] = value
    
    # sortierte artist_list, in der die ersten eintraege die sind, die bei meinen nachbarn am oeftesten vorkommen
    artist_list = sorted(artist_dictionary, key=artist_dictionary.get, reverse=True)
    # print "artist_dictionary"
    # print artist_dictionary

    # print "hoechster artist-value "
    # print artist_dictionary[artist_list[0]]

    recommended_artist_of_multiple_neighbours = artist_list[0:10]

    # print "recommended artist of multiple neighbours"
    # print recommended_artist_of_multiple_neighbours

    return recommended_artist_of_multiple_neighbours

    
    
def simple_neighbour(sort_idx, copy_UAM):
    neighbor_idx = sort_idx[-2]

    artists_of_neighbour = copy_UAM[neighbor_idx, :]
    # print "artists of neighbour"
    # print artists_of_neighbour

    artist_list = sorted(artists_of_neighbour, reverse=True)
    recommended_artist_of_simple_neighbour = artist_list[0:10]
    

    # print "recommended artist of simple neighbour: "
    # print recommended_artist_of_simple_neighbour
    return recommended_artist_of_simple_neighbour



#recommend_CF(5)

