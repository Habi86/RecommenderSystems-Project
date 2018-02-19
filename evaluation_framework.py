# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values
from collections import defaultdict
import baseline_recommenders
import collaborative_filtering
import popularity_based_recommender
import content_based_recommender
import hybrid_CF_PB
import hybrid_CF_CB
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM


NF = 10              # number of folds to perform in cross-validation
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)


amount_users = UAM.shape[0]
amount_artists = UAM.shape[1]

K = 10 # number of neighbours
# recommended_items_list = [5, 10, 15, 25, 50, 75, 100, 500]
recommended_items_list = range(10, 200, 10)
def evaluation_framework(method):
    prec_array = []
    rec_array = []
    f1_array = []
    tp_array = []
    #sample_users = random.sample(range(0, UAM.shape[0]), 15)
    sample_users =     range(20, 35)

    for number_recommended_items in recommended_items_list:

        avg_precision = 0.0       # mean precision
        avg_recall = 0.0        # mean recall
        tp = 0
        
        for user in sample_users:
            print "number recommended items: "
            print number_recommended_items
            print "user: "
            print user

            # Get seed user's artists listened to
            # user_row = UAM[user,:] # len = 10122
            user_row = np.nonzero(UAM[user, :])[0] # len = variabel
            # print "len user_row: "
            # print user_row

            if len(user_row)< 10: continue

            # kf = KFold(n_splits=NF)
            # for train, test in kf.split(user_row):
            folds = cross_validation.KFold(len(user_row), n_folds=NF)
            for train, test in folds:
            
                # np.set_printoptions(threshold=np.nan)
                # print "test: "
                # print test
                # print "user_row: "
                # print user_row
                # print len(user_row)
                # print len(test)
                # print "user_row[test]"
                # print user_row[test]
                

                train_UAM = UAM.copy()
                train_UAM[user, test] = 0.0

                # print (len(UAM[user, :]) - len(user_row))
                # print len(np.where(UAM[user, :] == 0)[0])

                # print "train_uam: "
                # print len(np.where(train_UAM[user, :] == 0)[0])
                


                if method == "CF":
                    # try: 
                    recommended_artists = collaborative_filtering.recommend_CF(user, train_UAM, K, number_recommended_items)
                    # except IndexError:
                    #     recommended_artists = []
                    #     continue
                elif method == "PB":
                    recommended_artists = popularity_based_recommender.recommend_PB(train_UAM, user, number_recommended_items)
                elif method == "CF_PB":
                    recommended_artists = hybrid_CF_PB.recommend_CF_PB(user, train_UAM, K, number_recommended_items)
                elif method == "RB_A":
                    recommended_artists = baseline_recommenders.recommend_RB_artist(train_UAM, user, number_recommended_items)
                elif method == "RB_U":
                    recommended_artists = baseline_recommenders.recommend_RB_user(user, train_UAM, number_recommended_items, K)
                elif method == "CB":
                    recommended_artists = content_based_recommender.recommend_CB(user_row[train], K, number_recommended_items)
                elif method == "CF_CB":
                    recommended_artists = hybrid_CF_CB.recommend_CF_CB(user, user_row[train], amount_artists, train_UAM, K, number_recommended_items)



                # print recommended_artists
                recommended_artists = np.array(recommended_artists)
                # print recommended_artists

                # print np.nonzero(UAM[user, :])[0]
                # print user_row[test]
                # raise "x"

                

                # print "recommended_artists: "
                # print recommended_artists

                # print "UAM test: "
                # print UAM[user, test]

                # print "len user_row: "
                # print len(user_row)
                # print "user row: "
                # print user_row
                # print "train: "
                # print train
                # print "test: "
                # print test
                # print "user_row[test]"
                # print user_row[test]


                
                correct_predicted_artists = np.intersect1d(user_row[test], recommended_artists)
                # correct_predicted_artists = np.intersect1d(train_UAM[user, test], recommended_artists)


                # print recommended_artists
                # print user_row[test]
                # print np.nonzero(UAM[user, test])[0]


                true_positives = len(correct_predicted_artists)
                tp = tp + true_positives
                

                # print "true positives: "
                # print true_positives
                # print len(recommended_artists)

                # wenn kein einziger artist empfohlen wird, precision = 100%
                if(len(recommended_artists) == 0):
                    precision = 100.0
                else:
                    precision = 100.0 * true_positives / len(recommended_artists)

                # print "precision: "
                # print precision
                # wenn kein einziger artist im test set vorkommt, recall = 100%
                if(len(test) == 0):
                    recall = 100.0
                else:
                    recall = 100.0 * true_positives / len(test)
                # print "len(test): "
                # print len(test)
                # print "recall: "
                # print recall
                # add precision and recall for current user and fold to aggregate variables
                avg_precision += precision / (NF * len(sample_users))
                avg_recall += recall / (NF * len(sample_users))

        
        if (avg_precision + avg_recall) != 0: f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
        else: f1_measure = 0.0
        f1_array.append(f1_measure)
        rec_array.append(avg_recall)
        prec_array.append(avg_precision)
        tp_array.append(tp)

        print "average precision: "
        print prec_array
        print "average recall: "
        print rec_array
        print "average f1: "
        print f1_array
        print "tp_array: "
        print tp_array

    
    np.savetxt('./plots/data/'+method+'_precision.txt', prec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_recall.txt', rec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_f1.txt', f1_array, delimiter=',')
    print "Done saving to file"


        


# plot_precision_recall()
evaluation_framework("CF_CB")


# CF:
# K = 3: 
# PRECISION LIST: 
# python evaluation_framework.py > ergebnisse_CF_5.txt

