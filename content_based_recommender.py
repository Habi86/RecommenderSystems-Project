# Content-based recommender:
# First, decide on one (or more) data source(s) to acquire external data on music items, e.g., web pages about the artists returned by a search engine, lyrics of the artists, or microblogs about the artists.
# Write a crawler that automatically fetches the respective ('music context') data for all artists in the collection.
# Then, create a representation of the artists inferred from your crawled data: e.g., term weight vectors according to the vector space model or co-occurrence information.
# Depending on your artist representation, choose a suited similarity measure and compute pairwise similarities between the artists:
# e.g., cosine similarity on term weight vectors, co-occurrence likelihood, or set-based Jaccard index.
# Finally, build a content-based recommender using the similarity matrix created.

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"       # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"           # user names for UAM
#AAM_FILE = ROOT_DIR + "AAM_100u.txt"                # artist-artist similarity matrix (AAM)
METHOD = "CB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]

NF = 10              # number of folds to perform in cross-validation #NEW default 2 auf 10
VERBOSE = False     # verbose output?


# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, K):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist


    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train,:], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:,-1-K:-1]


    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionary to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)
    for i in range(0, neighbor_idx.shape[0]):
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(neighbor_idx.flatten())     # First, we obtain a unique set of artists neighboring the seed user's artists.
    # Now, we find the positions of each unique neighbor in neighbor_idx.
    for nidx in uniq_neighbor_idx:
        mask = np.where(neighbor_idx == nidx)
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        # Store artist index and corresponding aggregated similarity in dictionary of arists to recommend
        dict_recommended_artists_idx[nidx] = avg_sim
    #########################################

    # Remove all artists that are in the training set of seed user
    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx, None)            # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx




# Function to run an evaluation experiment.
def run():
    # Initialize variables to hold performance measures
    avg_prec = 0   # mean precision
    avg_rec = 0    # mean recall
    f1_score = 0

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    for u in range(0, no_users):
        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        try:
            kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        except ValueError:
            NF < len(u_aidx)
            continue

        for train_aidx, test_aidx in kf:  # for all folds
            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable


            # Run recommendation method specified in METHOD
            # NB: u_aidx[train_aidx] gives the indices of training artists

            #K_RB = 10          # for RB: number of randomly selected artists to recommend
            #K_PB = 10          # for PB: number of most frequently played artists to recommend
            #K_CB = 3           # for CB: number of nearest neighbors to consider for each artist in seed user's training set
            #K_CF = 3           # for CF: number of nearest neighbors to consider for each user
            #K_HR = 10          # for hybrid: number of artists to recommend at most
            if METHOD == "RB":          # random baseline
                #dict_rec_aidx = recommend_RB_artist(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]), K_RB) # len(test_aidx))
                N = 100
                #dict_rec_aidx = recommend_RB_user(UAM, u_aidx[train_aidx], N, K_RB) # len(test_aidx))
            elif METHOD == "PB":        # popularity-based recommender
                print "PB"
                # dict_rec_aidx = recommend_PB(copy_UAM, u_aidx[train_aidx], K_PB) # len(test_aidx))
            elif METHOD == "CF":        # collaborative filtering
                print "PB"
                # dict_rec_aidx = recommend_CF(copy_UAM, u, u_aidx[train_aidx], K_CF)
            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], K_CB)

            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists
            # True Positives is amount of overlap in recommended artists and test artists
            TP = len(correct_aidx)
            # False Positives is recommended artists minus correctly predicted ones
            FP = len(np.setdiff1d(rec_aidx, correct_aidx))

            # Precision is percentage of correctly predicted among predicted
            # Handle special case that not a single artist could be recommended -> by definition, precision = 100%
            if len(rec_aidx) == 0:
                prec = 100.0
            else:
                prec = 100.0 * TP / len(rec_aidx)

            # Recall is percentage of correctly predicted among all listened to
            # Handle special case that there is no single artist in the test set -> by definition, recall = 100%
            if len(test_aidx) == 0:
                rec = 100.0
            else:
                rec = 100.0 * TP / len(test_aidx)


            # add precision and recall for current user and fold to aggregate variables
            avg_prec += prec / (NF * no_users)
            avg_rec += rec / (NF * no_users)
            f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec)) # NEW add F1 Score

            # Output precision and recall of current fold
            if VERBOSE:
                print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR  %.2f, F1: %.2f" % (avg_prec, avg_rec, f1_score))


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    #artists = read_from_file(ARTISTS_FILE)
    #users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    # Load AAM
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    # Number of neighbors / items to consider
    K_CB = 10
    K_CF = 10
    K_PB = 10

    # Run different experiments
    if False:
        METHOD = "CB"
        print METHOD
        for K_CB in range(1, 11):
            print (str(K_CB) + ","),
            run()

    if False:
        METHOD = "CF"
        print METHOD
        for K_CF in range(1, 11):
            print (str(K_CF) + ","),
            run()

    if True:
        METHOD = "RB"
        print METHOD
        for K_RB in range(1, 11):
            print (str(K_RB) + ","),
            run()

    if False:
        METHOD = "PB"
        print METHOD
        for K_PB in range(1, 11):
            print (str(K_PB) + ","),
            run()

    # For hyrbid methods, set K_CB, K_CF, and K_PB to reasonable values.
    if False:
        METHOD = "HR_SCB"
        print METHOD
        for K_HR in range(1, 11):
#            K_CB = K_CF = K_PB = K_HR
            K_CB = K_CF = K_PB = 3
            print (str(K_HR) + ","),
            run()

    if False:
        METHOD = "HR_RB"
        print METHOD
        for K_HR in range(1, 11):
#            K_CB = K_CF = K_PB = K_HR
            K_CB = K_CF = K_PB = K_HR
            print (str(K_HR) + ","),
            run()




