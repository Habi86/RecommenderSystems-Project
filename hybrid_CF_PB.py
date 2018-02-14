import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
from operator import itemgetter  # for sorting dictionaries w.r.t. values
from collections import defaultdict

import collaborative_filtering
import popularity_based_recommender


def recommend_CF_PB(user, UAM, K, number_recommended_items):
  cf = collaborative_filtering.recommend_CF(user, UAM, K, 100)
  pb = popularity_based_recommender.recommend_PB(UAM, 100)

  ranked_indizes_dictionary = defaultdict(list)
  for key, value in enumerate(cf):
    pb_index = np.where(pb == value)[0]

    if (pb_index.size):
        pb_index = pb_index[0]
    else: pb_index = len(pb)

    ranked_indizes_dictionary[value] = pb_index + key
    
    ranked_indizes = sorted(ranked_indizes_dictionary, key=ranked_indizes_dictionary.get, reverse=True)
    
  return ranked_indizes[0:number_recommended_items]


