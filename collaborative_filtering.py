# Load required modules
import numpy as np
from collections import defaultdict
import scipy.spatial.distance as scidist        # import distance computation module from scipy package



def recommend_CF(user, UAM, K, number_recommended_items):
    artists_of_user = UAM[user, :]
    similar_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        similar_users[u] = 1.0 - scidist.cosine(artists_of_user, UAM[u,:])
    sorted_similar_users = np.argsort(similar_users)

    neighbor_idx = sorted_similar_users[-1-K:-1]
    neighbours = UAM[neighbor_idx, :]

    artist_dictionary = defaultdict(list)
    for rank_index, neighbor_row in enumerate(neighbours):
        for key, playcount in enumerate(neighbor_row):
            if(playcount == 0): continue
            if(artist_dictionary[key]):
                artist_dictionary[key] = artist_dictionary[key] + (playcount*rank_index)
            else:
                artist_dictionary[key] = (playcount*rank_index)
    # sortierte artist_list, in der die ersten eintraege die sind, die bei meinen nachbarn am oeftesten vorkommen
    artist_list = sorted(artist_dictionary, key=artist_dictionary.get, reverse=True)


    recommended_artists = artist_list[0:number_recommended_items]
    return recommended_artists


# def multiple_neighbours(sorted_similar_users, K, UAM, number_recommended_items):

    
#     artist_dictionary = defaultdict(list)
#     for rank_index, neighbor_row in enumerate(neighbours):
#         for key, playcount in enumerate(neighbor_row):
#             if(playcount == 0): continue
#             if(artist_dictionary[key]):
#                 artist_dictionary[key] = artist_dictionary[key] + (playcount*rank_index)
#             else:
#                 artist_dictionary[key] = (playcount*rank_index)

#     # sortierte artist_list, in der die ersten eintraege die sind, die bei meinen nachbarn am oeftesten vorkommen
#     artist_list = sorted(artist_dictionary, key=artist_dictionary.get, reverse=True)
#     recommended_artists_of_multiple_neighbours = artist_list[0:number_recommended_items]
#     return recommended_artists_of_multiple_neighbours

    
    
# def simple_neighbour(sort_idx, UAM, number_recommended_items):
#     neighbor_idx = sort_idx[-2]
#     artists_of_neighbour = UAM[neighbor_idx, :]
#     artist_list = sorted(artists_of_neighbour, reverse=True)
#     recommended_artist_of_simple_neighbour = artist_list[0:number_recommended_items]
#     return recommended_artist_of_simple_neighbour


