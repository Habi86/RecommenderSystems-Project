# Content-based recommender:
# First, decide on one (or more) data source(s) to acquire external data on music items, e.g., web pages about the artists returned by a search engine, lyrics of the artists, or microblogs about the artists.
# Write a crawler that automatically fetches the respective ('music context') data for all artists in the collection.
# Then, create a representation of the artists inferred from your crawled data: e.g., term weight vectors according to the vector space model or co-occurrence information.
# Depending on your artist representation, choose a suited similarity measure and compute pairwise similarities between the artists:
# e.g., cosine similarity on term weight vectors, co-occurrence likelihood, or set-based Jaccard index.
# Finally, build a content-based recommender using the similarity matrix created.

