from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
data = pd.read_csv("netflix_titles.csv")

#removing stopwords
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')

#Replace NaN with an empty string
data['description'] = data['description'].fillna('')
data['title'] = data['title'].str.lower()

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#accessing title in this series
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
#print(indices)

# defining a function that recommends 10 most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend", methods = ["POST", "GET"])
def recommend():
    if request.method == "POST":
        movie = request.form['movie']
        movie = movie.lower()
        if movie in data['title'].values:
            r = get_recommendations(movie)
            r = r.str.title()
            return render_template('recommend.html', movie = movie.title(), r = r, t = 'yes')
        else:
            return render_template('recommend.html', movie = movie.title(), t = 'no')
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
