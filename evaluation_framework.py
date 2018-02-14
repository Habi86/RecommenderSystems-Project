

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values
from collections import defaultdict
import collaborative_filtering
import popularity_based_recommender
from pprint import pprint

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM.txt"                # artist-artist similarity matrix (AAM)


NF = 10              # number of folds to perform in cross-validation
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)


K = 10 # number of neighbours
recommended_items_list = [5, 10, 15, 25, 50, 75, 100, 500]
def evaluation_framework(method):
    prec_array = []
    rec_array = []
    sample_users = random.sample(range(0, UAM.shape[0]), 15)
    for number_recommended_items in recommended_items_list:
        
        avg_precision = 0;       # mean precision
        avg_recall = 0;        # mean recall
        
        for user in sample_users:
            print "user: "
            print user
            # Get seed user's artists listened to
            user_row = np.nonzero(UAM[user, :])[0]
            # user_row = UAM[user,:]

            
            # create folds
            folds = cross_validation.KFold(len(user_row), n_folds=NF)

            # split into train and test set
            for train, test in folds:

                train_UAM = UAM.copy()
                train_UAM[user, test] = 0.0

                if method == "CF":
                    # try: 
                    recommended_artists = collaborative_filtering.recommend_CF(user, train_UAM, K, number_recommended_items)
                    # except IndexError:
                    #     recommended_artists = []
                    #     continue
                elif method == "PB":
                    recommended_artists = popularity_based_recommender.recommend_PB(train_UAM, number_recommended_items)

                
                correct_predicted_artists = np.intersect1d(test, recommended_artists)

                true_positives = len(correct_predicted_artists)

                print "true positives: "
                print true_positives
                
                # wenn kein einziger artist empfohlen wird, precision = 100%
                if(len(recommended_artists) == 0):
                    precision = 100.0
                else:
                    precision = 100.0 * true_positives / len(recommended_artists)

                print "precision: "
                print precision
                # wenn kein einziger artist im test set vorkommt, recall = 100%
                if(len(test) == 0):
                    recall = 100.0
                else:
                    recall = 100.0 * true_positives / len(test)
                print "recall: "
                print recall
                # add precision and recall for current user and fold to aggregate variables
                avg_precision += precision / (NF * len(sample_users))
                avg_recall += recall / (NF * len(sample_users))

        print avg_precision
        print avg_recall
        f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
        rec_array.append(avg_recall)
        prec_array.append(avg_precision)

        print "average precision array: "
        print prec_array
        print "average recall array: "
        print rec_array
        


evaluation_framework("CF")




# CF:
# K = 3: 
# PRECISION LIST: 
# python evaluation_framework.py > ergebnisse_CF_5.txt