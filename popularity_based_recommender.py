# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).


# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
from operator import itemgetter  # for sorting dictionaries w.r.t. values

def recommend_PB(UAM, number_of_recommendations):   

    # amount_artists = UAM.shape[1]
    # 1 = spalten

    # if number_of_recommendations > amount_artists - len(train):
    #     number_of_recommendations = amount_artists - len(train)
    
    listening_events_per_artist = np.sum(UAM, axis=0)                           
    recommended_artists_idx = np.argsort(listening_events_per_artist) 
    top_recommendations = recommended_artists_idx[-number_of_recommendations:]

    return top_recommendations

# recommend_PB()