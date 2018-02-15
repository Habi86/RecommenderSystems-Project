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
import hybrid_CF_PB
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM.txt"                # artist-artist similarity matrix (AAM)


NF = 10              # number of folds to perform in cross-validation
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
amount_users = UAM.shape[0]
amount_artists = UAM.shape[1]

K = 20 # number of neighbours
# recommended_items_list = [5, 10, 15, 25, 50, 75, 100, 500]
recommended_items_list = range(0, 500, 10)
def evaluation_framework(method):
    prec_array = []
    rec_array = []
    f1_array = []
    # sample_users = random.sample(range(0, UAM.shape[0]), 15)
    sample_users = range(20, 25)


    for number_recommended_items in recommended_items_list:
        
        avg_precision = 0.0       # mean precision
        avg_recall = 0.0        # mean recall
        
        for user in sample_users:
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

                if method == "CF":
                    # try: 
                    recommended_artists = collaborative_filtering.recommend_CF(user, train_UAM, K, number_recommended_items)
                    # except IndexError:
                    #     recommended_artists = []
                    #     continue
                elif method == "PB":
                    recommended_artists = popularity_based_recommender.recommend_PB(train_UAM, number_recommended_items)
                elif method == "CF_PB":
                    recommended_artists = hybrid_CF_PB.recommend_CF_PB(user, train_UAM, K, number_recommended_items)

                elif method == "RB_A":
                    recommended_artists = baseline_recommenders.recommend_RB_artist(np.setdiff1d(range(0, amount_artists), user_row[train]), K)

                elif method == "RB_U":
                    N = 100
                    recommended_artists = baseline_recommenders.recommend_RB_user(train_UAM, user_row[train], N, K)



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
                
                correct_predicted_artists = np.intersect1d(user_row[test], recommended_artists)

                true_positives = len(correct_predicted_artists)

                print "true positives: "
                print true_positives
                print len(recommended_artists)
                # raise "x"
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
                print "len(test): "
                print len(test)
                print "recall: "
                print recall
                # add precision and recall for current user and fold to aggregate variables
                avg_precision += precision / (NF * len(sample_users))
                avg_recall += recall / (NF * len(sample_users))

        
        f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
        f1_array.append(f1_measure)
        rec_array.append(avg_recall)
        prec_array.append(avg_precision)

        print "average precision: "
        print prec_array
        print "average recall: "
        print rec_array
        print "f1: "
        print f1_array
        # print "average f1: "
        # print f1_measure
    
    np.savetxt('./plots/data/'+method+'_precision.txt', prec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_recall.txt', rec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_f1.txt', f1_array, delimiter=',')
    # print np.loadtxt('./plots/data/cf-precision.txt', delimiter=',')


        




# plot_precision_recall()
evaluation_framework("CF")





# CF:
# K = 3: 
# PRECISION LIST: 
# python evaluation_framework.py > ergebnisse_CF_5.txt