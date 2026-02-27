from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

app = Flask(__name__)

# Load data
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

# Rename columns in df1 for merging
df1.columns = ['id', 'title', 'cast', 'crew']

# Merge datasets
df2 = df2.merge(df1, on='id')

# Handle duplicated title columns from the merge
df2['title'] = df2['title_y']
df2 = df2.drop(['title_x', 'title_y'], axis=1)

# Preprocess for content-based filtering
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

df2['director'] = df2['crew'].apply(get_director)

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names
    return []

for feature in ['cast', 'keywords', 'genres']:
    df2[feature] = df2[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ''

for feature in ['cast', 'keywords', 'director', 'genres']:
    df2[feature] = df2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)

# Generate the count matrix and cosine similarity matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices].tolist()

# API for auto-fill suggestions
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q')
    suggestions = df2[df2['title'].str.contains(search, case=False, na=False)]['title'].tolist()
    return jsonify(suggestions)

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = get_recommendations(movie_title.title())
        return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
