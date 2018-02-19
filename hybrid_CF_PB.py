import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
from operator import itemgetter  # for sorting dictionaries w.r.t. values
from collections import defaultdict

import collaborative_filtering
import popularity_based_recommender


def recommend_CF_PB(user, UAM, K, number_recommended_items):
  cf_artist_indizes = collaborative_filtering.recommend_CF(user, UAM, K, 100)
  pb_artist_indizes = popularity_based_recommender.recommend_PB(UAM, user, 100)

  ranked_indizes_dictionary = defaultdict(list)

  
  for cf_index, value in enumerate(cf_artist_indizes):
    pb_index = np.where(pb_artist_indizes == value)[0]
    if (pb_index.size):
        pb_index = pb_index[0]
    else: pb_index = len(pb_artist_indizes)

    ranked_indizes_dictionary[value] = pb_index + cf_index
    
    ranked_indizes = sorted(ranked_indizes_dictionary, key=ranked_indizes_dictionary.get)

  return ranked_indizes[0:number_recommended_items]


