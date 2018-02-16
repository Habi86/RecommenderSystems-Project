# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).


# Load required modules
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
from operator import itemgetter  # for sorting dictionaries w.r.t. values

def recommend_PB(UAM, user, number_of_recommendations):
    listening_events_per_artist = np.sum(UAM, axis=0)                           
    top_artists = np.argsort(listening_events_per_artist)

    recommended_artists = top_artists[-number_of_recommendations:]
    return recommended_artists[::-1]




    # user_row = UAM[user, :]
    # unknown_artists_of_user = np.where(user_row == 0)[0]
    # # print len(np.where(UAM[user, :] == 0)[0])
    # # print len(unknown_artists_of_user)
    # # raise "x"
    
    # listening_events_per_artist = np.sum(UAM, axis=0)             
    # top_artists_reversed = np.argsort(listening_events_per_artist)
    # top_artists = top_artists_reversed[::-1]

    # recommended_artists = []
    # for artist in top_artists:
    #     if artist in unknown_artists_of_user:
    #         recommended_artists.append(artist)

    #     if len(recommended_artists) == number_of_recommendations: break
    # return recommended_artists






    # return [42, 260, 217, 43, 590, 268, 19, 3788, 228, 2792]




    # return [42, 260, 217, 43, 590, 268, 19, 3788, 228, 2792]
    # return [  42  260  217   43  590  268   19 3788  228 2792]



#  [42, 260, 217, 590, 19, 3788, 2792, 298, 20, 2646] nur unknown user
#  [  42  260  217   43  590  268   19 3788  228 2792]