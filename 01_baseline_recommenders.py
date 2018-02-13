# Baseline recommenders:
# First, implement two simple baseline algorithms against which you compare your more sophisticated recommenders:
# one that recommends randomly selected artists the target user has not listened to before and
# one that recommends artists listened to by randomly selected users
# (of course, excluding artists already known by the target user, i.e. artists in the 'training set').
# Implement them as functions recommend_RB_artist and recommend_RB_user.


# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values

# Parameters
ROOT_DIR = "./data/"
UAM_FILE = ROOT_DIR + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "LFM1b_artists.txt"       # artist names for UAM
USERS_FILE = ROOT_DIR + "LFM1b_users.txt"           # user names for UAM
METHOD = "RB"                                       # recommendation method
                                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]

NF = 2               # number of folds to perform in cross-validation
VERBOSE = False      # verbose output?

# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        reader.next()  # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB_artist(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx

# Function that implements a dumb random recommender. It predicts a number of artists from randomly chosen users.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB_user(UAM, seed_aidx_train, no_items, K_users = 1):
    # UAM                   user-artist-matrix
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict
    # K_users               no of random users selected

    # Select a random sample of users
    random_uidx = random.sample(range(0,UAM.shape[0]), K_users)
    # Get artits of these
    random_aidx_nz = np.nonzero(UAM[random_uidx,:])[1]      # only interested in artists, hence [1]
    # Remove artists in training set of seed user
    random_aidx = np.setdiff1d(set(random_aidx_nz), seed_aidx_train)

    if VERBOSE:
        print str(K_users) + ' user(s) randomly chosen, ' + str(no_items) + ' recommendations requested, ' + str(len(random_aidx)) + ' found' # restart with K=' + str(K_users+1)

    # Start over with increased number of users to consider, if recommended artists smaller than requested
    if len(random_aidx) < no_items:
        K_users += 1
        return recommend_RB_user(UAM, seed_aidx_train, no_items, K_users)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx


# Function to run an evaluation experiment.
def run():
    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:  # for all folds
            # Show progress
            if VERBOSE :
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)) + '\n',      # the comma at the end avoids line break
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
#               dict_rec_aidx = recommend_RB(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]), K_RB) # len(test_aidx))
                N = 100
                dict_rec_aidx = recommend_RB_user(UAM, u_aidx[train_aidx], N, K_RB) # len(test_aidx))
                #...

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

            # Output precision and recall of current fold
            if VERBOSE:
                print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec) + "\n")

            # Increase fold counter
            fold += 1

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR  %.2f" % (avg_prec, avg_rec))
    print ("%.3f, %.3f" % (avg_prec, avg_rec))


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    # Load AAM
    #AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    # Number of neighbors / items to consider
    K_CB = 10
    K_CF = 10
    K_PB = 10

    # Run different experiments
    if True:
        METHOD = "RB"
        print METHOD
        for K_RB in range(1, 11):
            print (str(K_RB) + ","),
            run()