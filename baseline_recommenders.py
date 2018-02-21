# Load required modules
import numpy as np
import random

def recommend_RB_artist(UAM, user, number_recommended_items):
    user_row = UAM[user, :]
    unknown_artists_of_user = np.where(user_row == 0)[0]
    recommended_items = random.sample(unknown_artists_of_user, number_recommended_items)
    return recommended_items



def recommend_RB_user(user, UAM, number_recommended_items, K):
    random_users = random.sample(range(0,UAM.shape[0]), K)
    artists_of_random_users = np.nonzero(UAM[random_users,:])[1]

    unique_artists_of_random_users = []
    for artist in artists_of_random_users:
        if artist not in unique_artists_of_random_users:
            unique_artists_of_random_users.append(artist)

    recommended_items = random.sample(unique_artists_of_random_users, number_recommended_items)

    return recommended_items