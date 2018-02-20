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

#sample_users = random.sample(range(0, UAM.shape[0]), 15)
sample_users = [210, 522, 235, 207, 475, 76, 650, 362, 227, 582, 396, 1052, 492, 1032, 751]

K = 10 # number of neighbours
# recommended_items_list = [5, 10, 15, 25, 50, 75, 100, 500]
recommended_items_list = range(10, 200, 10)
def evaluation_framework(method):
    prec_array = []
    rec_array = []
    f1_array = []
    tp_array = []

    for number_recommended_items in recommended_items_list:

        avg_precision = 0.0       # mean precision
        avg_recall = 0.0        # mean recall
        tp = 0
        
        for user in sample_users:
            # print "number recommended items: "
            # print number_recommended_items
            # print "user: "
            # print user

            # Get seed user's artists listened to
            # user_row = UAM[user,:] # len = 10122
            user_row = np.nonzero(UAM[user, :])[0] # len = variabel
            # print "len user_row: "
            # print user_row

            if len(user_row)< K: continue

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

        # print "average precision: "
        # print prec_array
        # print "average recall: "
        # print rec_array
        # print "average f1: "
        # print f1_array
        # print "tp_array: "
        # print tp_array

    
    np.savetxt('./plots/data/'+method+'_precision.txt', prec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_recall.txt', rec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_f1.txt', f1_array, delimiter=',')
    print "Done saving to file"
    print method



def cold_start_evaluation(method):
    number_recommended_items = 50
    # prec_array = []
    # rec_array = []
    f1_array = []
    user_playcounts_array = []

    avg_precision = 0.0       # mean precision
    avg_recall = 0.0        # mean recall
    avg_user_playcount = 0
    user_count = 0
    summed_user_playcounts = 0
    # f1_measure = 0.0

    summed_user_playcounts = np.sum(UAM, axis=1)
    sorted_user_indizes = np.argsort(summed_user_playcounts)

   

    for user in sorted_user_indizes:
        # print "user: "
        # print user
        
        print "playcounts array len: "
        print len(user_playcounts_array)

        user_playcount = np.sum(UAM[user, :])

        # print "playcounts: "
        print user_playcount

        user_row = np.nonzero(UAM[user, :])[0]

        if len(user_row) < K: continue

        folds = cross_validation.KFold(len(user_row), n_folds=NF)
        for train, test in folds:

            train_UAM = UAM.copy()
            train_UAM[user, test] = 0.0
            

            if method == "CF":
                recommended_artists = collaborative_filtering.recommend_CF(user, train_UAM, K, number_recommended_items)
            elif method == "PB":
                recommended_artists = popularity_based_recommender.recommend_PB(train_UAM, user, number_recommended_items)
            elif method == "CF_PB":
                recommended_artists = hybrid_CF_PB.recommend_CF_PB(user, train_UAM, K, number_recommended_items)
            elif method == "RB_A":
                recommended_artists = baseline_recommenders.recommend_RB_artist(train_UAM, user, number_recommended_items)

            elif method == "RB_U":
                recommended_artists = baseline_recommenders.recommend_RB_user(user, train_UAM, number_recommended_items, K)
            elif method == "CB":
                # recommended_artists = content_based_recommender.recommend_CB(user, train_UAM,  number_recommended_items, K)
                recommended_artists = content_based_recommender.recommend_CB(user_row[train], K, number_recommended_items)
            elif method == "CF_CB":
                recommended_artists = hybrid_CF_CB.recommend_CF_CB(user, user_row[train], amount_artists, train_UAM, K, number_recommended_items)

            # recommended_artists = np.array(recommended_artists)
            correct_predicted_artists = np.intersect1d(user_row[test], recommended_artists)
            true_positives = len(correct_predicted_artists)
            # tp = tp + true_positives

            # wenn kein einziger artist empfohlen wird, precision = 100%
            if(len(recommended_artists) == 0):
                precision = 100.0
            else:
                precision = 100.0 * true_positives / len(recommended_artists)

            # wenn kein einziger artist im test set vorkommt, recall = 100%
            if(len(test) == 0):
                recall = 100.0
            else:
                recall = 100.0 * true_positives / len(test)

            # add precision and recall for current user and fold to aggregate variables
            avg_precision += precision / (NF)
            avg_recall += recall / (NF)
            summed_user_playcounts += user_playcount


        user_count += 1
        if user_playcount > (20000 * (len(user_playcounts_array)+1)):
            avg_precision = avg_precision / user_count
            avg_recall = avg_recall / user_count
            if (avg_precision + avg_recall) != 0: f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
            else: f1_measure = 0.0

            user_playcounts_array.append(user_playcount / user_count)
            f1_array.append(f1_measure)
            summed_user_playcounts = 0
            user_count = 0
            avg_precision = 0.0
            avg_recall = 0.0
            f1_measure = 0.0

    
    np.savetxt('./plots/data/cold-start/'+method+'_f1.txt', f1_array, delimiter=',')
    np.savetxt('./plots/data/cold-start/user_playcounts.txt', user_playcounts_array, delimiter=',')
    print "Done saving to file"




# plot_precision_recall()
evaluation_framework("RB_A")
evaluation_framework("RB_U")
evaluation_framework("CF")
evaluation_framework("CB")
evaluation_framework("PB")
evaluation_framework("CF_PB")
evaluation_framework("CF_CB")

#cold_start_evaluation("CB")


print("######## SAMPLE USER")
print(sample_users)



