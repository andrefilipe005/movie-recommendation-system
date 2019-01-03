import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings; warnings.simplefilter('ignore')


md = pd.read_csv('data/movies_metadata.csv', low_memory=False)
links_small = pd.read_csv('data/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

smd = md[md['id'].isin(links_small)]

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

def get_recommendations(title,cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return smd.iloc[movie_indices][['title', 'vote_count', 'vote_average']]

#print(get_recommendations('The Godfather',cosine_sim).head(10))
