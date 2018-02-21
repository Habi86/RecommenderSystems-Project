# Load required modules
import numpy as np

ROOT_DIR = "./data/"
AAM_FILE = ROOT_DIR + "wikipedia/AAM.txt"
AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)                

def recommend_CB(artist_indizes, K, number_of_recommendations):

    sort_artist_indizes = np.argsort(AAM[artist_indizes, :], axis=1) # Get nearest neighbors of train set artist of seed user; Sort AAM column-wise for each row
    similar_neighbor_artists_indizes = sort_artist_indizes[:,-1-K:-1] # Select the K closest artists to all artists the seed user listened to

    dict_recommended_artists_idx = {}
    sims_neighbors_idx = np.zeros(shape=(len(artist_indizes), K), dtype=np.float32)     # Distill corresponding similarity scores and store in similar_neighbor_artists_indizes
    for i in range(0, similar_neighbor_artists_indizes.shape[0]):
        sims_neighbors_idx[i] = AAM[artist_indizes[i], similar_neighbor_artists_indizes[i]]

    # Aggregate the artists in similar_neighbor_indizes.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(similar_neighbor_artists_indizes.flatten())     # First, we obtain a unique set of artists neighboring the seed user's artists.
    # Now, we find the positions of each unique neighbor in similar_neighbor_indizes.
    for nidx in uniq_neighbor_idx:
        mask = np.where(similar_neighbor_artists_indizes == nidx)
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        # Store artist index and corresponding aggregated similarity in dictionary of arists to recommend
        dict_recommended_artists_idx[nidx] = avg_sim


    # Remove all artists that are in the training set of seed user
    for aidx in artist_indizes:
        dict_recommended_artists_idx.pop(aidx, None)   # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    recommended_artists_idx = sorted(dict_recommended_artists_idx, key=dict_recommended_artists_idx.get, reverse=True)
    recommended_items = recommended_artists_idx[0:number_of_recommendations]

    return recommended_items
