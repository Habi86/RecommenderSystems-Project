## Recommender Systems: Final Project

### In carrying out your recommender systems project, address the following tasks:
**Baseline recommenders:**<br>
First, implement two simple baseline algorithms against which you compare your more sophisticated recommenders: one that recommends randomly selected artists the target user has not listened to before and one that recommends artists listened to by randomly selected users (of course, excluding artists already known by the target user, i.e. artists in the “training set”). Implement them as functions recommend_RB_artist and recommend_RB_user.
Collaborative filtering:
Implement a user-based, memory-based collaborative filtering artist recommender that supports k- nearest neighbors prediction. In addition to the version we implemented in the lab, elaborate and report a method to combine the predictions for the same artists among the set of nearest neighbors (e.g., how to deal with an artist that is recommended 10 times by 20 nearest neighbors vs. an artist that is recommended only once, but by a neighbor with a music taste very similar to the target user; think of combining user similarity and artist frequency). Implement your recommender in a function recommend_CF.

**Content-based recommender:**<br>
First, decide on one (or more) data source(s) to acquire external data on music items, e.g., web pages about the artists returned by a search engine, lyrics of the artists, or microblogs about the artists. Write a crawler that automatically fetches the respective (“music context”) data for all artists in the collection.
Then, create a representation of the artists inferred from your crawled data: e.g., term weight vectors according to the vector space model or co-occurrence information.
Depending on your artist representation, choose a suited similarity measure and compute pairwise similarities between the artists: e.g., cosine similarity on term weight vectors, co-occurrence likelihood, or set-based Jaccard index.
Finally, build a content-based recommender using the similarity matrix created.

**Popularity-based recommender:**<br>
Implement a simple recommender (recommend_PB) that always predicts the overall most popular artists in the dataset, irrespective of a particular user profile and item content.

**Hybrid recommender:**<br>
Implement at least one way to integrate CF, CB, and PB to build a hybrid recommender (e.g., rank- based or set-based fusion). Evaluate your method at least on the combinations CF+CB and CF+PB.

### Evaluation:
Build an evaluation framework to perform cross-fold validation on the user level. Evaluate all your recommendation algorithms on the dataset C1ku (from MediaCube) using 10-fold cross-validation (CV), i.e., for each user, split their unique artists into 90% training artists and 10% testing artists and iterate 10 times to cover all possible 9:1 splits. Compute basic figures of merit, at least (average) precision, recall, and F1 measure. <br>
Investigate how results change for different numbers of neighbors k and different numbers of predicted artists n.
Create precision/recall plots (such as the one shown in precision_recall_plot.pdf on MediaCube) that allow to easily analyze the trade-off between recall and precision for the different approaches. In addition, you should investigate and discuss in a structured and comprehensive manner the relationship between the performance measures and number of recommended items.<br>
Ensure (implementation-wise) that the investigated approaches recommend the same number of items in order to compare them in a fair way.
Furthermore, investigate cold-start scenarios, i.e. analyze the influence of user activity (e.g. number of listening events) of the target user on the performance of your recommenders. <br>
To this end, you may sort users according to their listening intensity and plot performance measures such as R- precision or F1 measure against listening intensity (e.g., using a scatter plot). Do this for all recommendation approaches. For which approaches are you able to identify a relationship between user activity and recommendation performance?