

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

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM.txt"                # artist-artist similarity matrix (AAM)
METHOD = "PB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]
NF = 10              # number of folds to perform in cross-validation
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
K = 3
def evaluation_framework(M):
    METHOD = M
    precision_list = []
    recall_list = []
    avg_precision = 0;       # mean precision
    avg_recall = 0;        # mean recall

    # amount_user = 1
    amount_user = UAM.shape[0]
    print "amount_user: "
    print amount_user
    amount_artist = UAM.shape[1]
    for user in range(0, amount_user):
        print "user: "
        print user
        # Get seed user's artists listened to
        users_artists = np.nonzero(UAM[user, :])[0]


        # split into train and test set
        # create folds
        # print " users artists: "
        # print len(users_artists)
        try:
            folds = cross_validation.KFold(len(users_artists), n_folds=NF)
        except ValueError:
            NF < len(users_artists)
            continue
        
        for train, test in folds:
            copy_UAM = UAM.copy()


            if METHOD == "CF":
                try: 
                    recommended_artists = collaborative_filtering.recommend_CF(user, copy_UAM, users_artists[train], K)
                except IndexError:
                    recommended_artists = []
                    continue
            elif METHOD == "PB":
                recommended_artists = popularity_based_recommender.recommend_PB(copy_UAM, users_artists[train], K)
              

            #recommended_artists = dict_recommended_artists.keys
            # print "recommended items: "
            # print recommended_artists

            try:
                correct_predicted_artists = np.intersect1d(users_artists[test], recommended_artists)
            except TypeError:
                len(users_artists) < 2
                continue
            
            true_positives = len(correct_predicted_artists)
            false_positives = len(np.setdiff1d(recommended_artists, correct_predicted_artists))

            # wenn kein einziger artist empfohlen wird, precision = 100%
            if(len(recommended_artists) == 0):
                precision = 100.0
            else:
                precision = 100.0 * true_positives / len(recommended_artists)

            
            precision_list.append(precision)
            
            
            # wenn kein einziger artist im test set vorkommt, recall = 100%
            if(len(test) == 0):
                recall = 100.0
            else:
                recall = 100.0 * true_positives / len(test)

            recall_list.append(recall)
            

            # add precision and recall for current user and fold to aggregate variables
            avg_precision += precision / (NF * amount_user)
            avg_recall += recall / (NF * amount_user)

            print "average precision: "
            print avg_precision

            print "average recall: "
            print avg_recall


    print "average precision & recall: "
    print ("%.3f, %.3f" % (avg_precision, avg_recall))
    print "recall List: "
    print recall_list
    print "precision List: "
    print precision_list


evaluation_framework("PB")




# CF:
# K = 3: 
# PRECISION LIST: 
# 