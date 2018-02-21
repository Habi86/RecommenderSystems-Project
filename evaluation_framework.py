# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
from sklearn.model_selection import KFold
import random
from collections import defaultdict
import scipy.spatial.distance as scidist        # import distance computation module from scipy package

import baseline_recommenders
import collaborative_filtering
import popularity_based_recommender
import content_based_recommender
import hybrid_CF_PB
import hybrid_CF_CB

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"       # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"           # user names for UAM


NF = 10                                             # number of folds to perform in cross-validation
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

amount_artists = UAM.shape[1]

sample_users = random.sample(range(0, UAM.shape[0]), 15)
#sample_users = [210, 522, 235, 207, 475, 76, 650, 362, 227, 582, 396, 1052, 492, 1032, 751] # = random users with best result

K = 10                                              # number of neighbours
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

            user_row = np.nonzero(UAM[user, :])[0] # len = variabel

            if len(user_row)< K: continue

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
                    recommended_artists = content_based_recommender.recommend_CB(user_row[train], K, number_recommended_items)
                elif method == "CF_CB":
                    recommended_artists = hybrid_CF_CB.recommend_CF_CB(user, user_row[train], amount_artists, train_UAM, K, number_recommended_items)

                correct_predicted_artists = np.intersect1d(user_row[test], recommended_artists)
                true_positives = len(correct_predicted_artists)
                tp = tp + true_positives

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
                avg_precision += precision / (NF * len(sample_users))
                avg_recall += recall / (NF * len(sample_users))

        if (avg_precision + avg_recall) != 0: f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
        else: f1_measure = 0.0
        f1_array.append(f1_measure)
        rec_array.append(avg_recall)
        prec_array.append(avg_precision)
        tp_array.append(tp)
    
    np.savetxt('./plots/data/'+method+'_precision.txt', prec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_recall.txt', rec_array, delimiter=',')
    np.savetxt('./plots/data/'+method+'_f1.txt', f1_array, delimiter=',')
    print "Done saving to file"
    print method


def cold_start_evaluation(method):
    number_recommended_items = 50
    f1_array = []
    user_playcounts_array = []

    avg_precision = 0.0       # mean precision
    avg_recall = 0.0          # mean recall
    avg_user_playcount = 0
    user_count = 0
    user_playcounts_sum = 0

    summed_user_playcounts = np.sum(UAM, axis=1)
    sorted_user_indizes = np.argsort(summed_user_playcounts)

    for user in sorted_user_indizes:

        user_playcount = np.sum(UAM[user, :])
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
                recommended_artists = content_based_recommender.recommend_CB(user_row[train], K, number_recommended_items)
            elif method == "CF_CB":
                recommended_artists = hybrid_CF_CB.recommend_CF_CB(user, user_row[train], amount_artists, train_UAM, K, number_recommended_items)

            correct_predicted_artists = np.intersect1d(user_row[test], recommended_artists)
            true_positives = len(correct_predicted_artists)

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
            user_playcounts_sum += user_playcount


        user_count += 1
        if user_playcount > (10000 * (len(user_playcounts_array)+1)):
            avg_precision = avg_precision / user_count
            avg_recall = avg_recall / user_count
            if (avg_precision + avg_recall) != 0: f1_measure = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
            else: f1_measure = 0.0

            user_playcounts_array.append(user_playcounts_sum / user_count)
            f1_array.append(f1_measure)
            user_playcounts_sum = 0
            user_count = 0
            avg_precision = 0.0
            avg_recall = 0.0
            f1_measure = 0.0

    print "user playcounts array: "
    print user_playcounts_array
    
    # np.savetxt('./plots/data/cold-start/10000/'+method+'_f1_neu.txt', f1_array, delimiter=',')
    np.savetxt('./plots/data/cold-start/10000/user_playcounts_02.txt', user_playcounts_array, delimiter=',')
    print "Done saving to file"



cold_start_evaluation("PB")
# evaluation_framework("RB_A")
# evaluation_framework("RB_U")
# evaluation_framework("CF")
# evaluation_framework("CB")
# evaluation_framework("PB")
# evaluation_framework("CF_PB")
# evaluation_framework("CF_CB")

#cold_start_evaluation("CB")

# print("######## SAMPLE USER")
# print(sample_users)



