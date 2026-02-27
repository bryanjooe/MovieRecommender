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
df2['title'] = df2['title_y']  # Use title_y as the main title column
df2 = df2.drop(['title_x', 'title_y'], axis=1)  # Remove redundant columns

# Calculate C and M for weighted ratings
C = df2['vote_average'].mean()
M = df2['vote_count'].quantile(0.9)
q_movies = df2.copy().loc[df2['vote_count'] >= M]

# Weighted Rating Function
def w_rating(x, m=M, c=C):
    v = x['vote_count']
    R = x['vote_average']
    return ((v / (v + m)) * R) + ((m / (v + m)) * C)

q_movies['score'] = q_movies.apply(w_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

# Content-Based Filtering Preprocessing
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
        return [i['name'] for i in x]
    return []

for feature in ['cast', 'keywords', 'genres']:
    df2[feature] = df2[feature].apply(get_list)

# Clean Data
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ''

for feature in ['cast', 'keywords', 'director', 'genres']:
    df2[feature] = df2[feature].apply(clean_data)

# Create Content 'Soup'
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)

# Cosine Similarity Matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices].tolist()

# === Routes ===

# Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = get_recommendations(movie_title.title())
        return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)
    return render_template('index.html')

# Auto-fill Feature
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q', '')  # Get search term from query
    suggestions = df2[df2['title'].str.contains(search, case=False, na=False)]['title'].tolist()[:10]
    return jsonify(suggestions)

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
