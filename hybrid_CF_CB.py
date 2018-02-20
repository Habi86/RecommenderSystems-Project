# Hybrid recommender:
# Implement at least one way to integrate CF, CB, and PB to build a hybrid recommender (e.g., rank- based or set-based fusion).
# Evaluate your method at least on the combinations CF+CB and CF+PB.

import numpy as np
from collections import defaultdict

import collaborative_filtering
import content_based_recommender


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


    # def recommend_CF_CB(user, artist_indizes, amount_artists, UAM, K, number_recommended_items):
#     dict_rec_aidx_CF = collaborative_filtering.recommend_CF(user, UAM, K, 100)
#     dict_rec_aidx_CB = content_based_recommender.recommend_CB(artist_indizes, K, 100)
#
#     print dict_rec_aidx_CB
#     print dict_rec_aidx_CF
#
#
#     # First, create matrix to hold scores per recommendation method per artist
#     scores = np.zeros(shape=(2, amount_artists), dtype=np.float32)
#     # Add scores from CB and CF recommenders to this matrix
#     for aidx in dict_rec_aidx_CB.keys():
#         scores[0, aidx] = dict_rec_aidx_CB[aidx]
#     for aidx in dict_rec_aidx_CF:
#         scores[1, aidx] = dict_rec_aidx_CF[aidx]
#
#     # Apply aggregation function (here, just take arithmetic mean of scores)
#     scores_fused = np.mean(scores, axis=0)
#     # Sort and select top K artists to recommend
#     sorted_idx = np.argsort(scores_fused)
#     sorted_idx_top = sorted_idx[-K:]
#     # Put (artist index, score) pairs of highest scoring artists in a dictionary
#     dict_rec_aidx = {}
#     for i in range(0, len(sorted_idx_top)):
#         dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]
#
#     recommended_items = dict_rec_aidx[0:number_recommended_items]
#
#     return recommended_items