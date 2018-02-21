import numpy as np
from collections import defaultdict
import collaborative_filtering
import content_based_recommender
import scipy.spatial.distance as scidist        # import distance computation module from scipy package


def recommend_CF_CB(user, artist_indizes, amount_artists, UAM, K, number_recommended_items):
    rec_aidx_CF = collaborative_filtering.recommend_CF(user, UAM, K, 100)
    rec_aidx_CB = content_based_recommender.recommend_CB(artist_indizes, K, 100)

    ranked_indizes_dictionary = defaultdict(list)

    for cf_index, value in enumerate(rec_aidx_CF):
        pb_index = np.where(rec_aidx_CB == value)[0]
        if (pb_index.size):
            pb_index = pb_index[0]
        else:
            pb_index = len(rec_aidx_CB)

        ranked_indizes_dictionary[value] = pb_index + cf_index

        ranked_indizes = sorted(ranked_indizes_dictionary, key=ranked_indizes_dictionary.get)

    return ranked_indizes[0:number_recommended_items]