# Load required modules
import numpy as np
import scipy.spatial.distance as scidist
from collections import defaultdict



def recommend_CF(user, UAM, K, number_recommended_items):
    artists_of_user = UAM[user, :]
   
    similar_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    
    # Compute similarities as inverse cosine distance between user_playcount of user and all users via UAM (assuming that UAM is normalized)
    for u in range(0, UAM.shape[0]):
        similar_users[u] = 1.0 - scidist.cosine(artists_of_user, UAM[u,:])
    np.set_printoptions(threshold=np.nan)
    # print similar_users
    # similarity der user absteigend sortieren
    sorted_similar_users = np.argsort(similar_users)
    # print "sorted_similar_users"
    # print sorted_similar_users
    
    recommended_artists = multiple_neighbours(sorted_similar_users, K, UAM, number_recommended_items)
    # recommended_artists = simple_neighbour(sorted_similar_users, UAM, number_recommended_items)
    return recommended_artists


def multiple_neighbours(sorted_similar_users, K, UAM, number_recommended_items):
    neighbor_idx = sorted_similar_users[-1-K:-1]
    # print "neighbor_idx"
    # print neighbor_idx
    
    neighbours = UAM[neighbor_idx, :]
    # print "neighbours"
    # print neighbours

    artist_dictionary = defaultdict(list)
    for rank_index, neighbor_row in enumerate(neighbours):
        for key, playcount in enumerate(neighbor_row):
            if(playcount == 0): continue
            if(artist_dictionary[key]):
                artist_dictionary[key] = artist_dictionary[key] + (playcount*rank_index)
            else:
                artist_dictionary[key] = (playcount*rank_index)
    
    # print "artist_dict: "
    # print artist_dictionary

    # sortierte artist_list, in der die ersten eintraege die sind, die bei meinen nachbarn am oeftesten vorkommen
    artist_list = sorted(artist_dictionary, key=artist_dictionary.get, reverse=True)

    # print "artist_list"
    # print artist_list
    
    # print "hoechster artist-value "
    # print artist_dictionary[artist_list[0]]

    recommended_artists_of_multiple_neighbours = artist_list[0:number_recommended_items]

    # print "recommended_artist_of_multiple_neighbours"
    # print recommended_artist_of_multiple_neighbours

    return recommended_artists_of_multiple_neighbours

    
    
def simple_neighbour(sort_idx, UAM, number_recommended_items):
    neighbor_idx = sort_idx[-2]

    artists_of_neighbour = UAM[neighbor_idx, :]
    # print "artists of neighbour"
    # print artists_of_neighbour

    artist_list = sorted(artists_of_neighbour, reverse=True)
    recommended_artist_of_simple_neighbour = artist_list[0:number_recommended_items]
    

    # print "recommended artist of simple neighbour: "
    # print recommended_artist_of_simple_neighbour
    return recommended_artist_of_simple_neighbour



#recommend_CF(5)

