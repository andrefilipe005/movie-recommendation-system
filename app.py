import rs.contentBased_option1 as cb1
import rs.contentBases_option2 as cb2
import rs.popularity as pop
import rs.collaborative_filtering_handler as cf
import warnings; warnings.simplefilter('ignore')

from flask import Flask,render_template, request

app = Flask(__name__)


@app.route('/',methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        popularity_movies = pop.popularity_by_genre(request.form['genre'],0.85).head(10)
    else:
        popularity_movies = pop.popularity(0.95).head(10)

    popularity_movies.set_index("title",inplace=True)
    return render_template("movies.html", popularity_movies = popularity_movies.index.values)

@app.route('/content_based_filtering',methods = ['POST', 'GET'])
def content_based():
   if request.method == 'POST':
      recommendations_description = cb1.get_recommendations(request.form['movie'],cb1.cosine_sim).head(10)
      recommendations_keywords = cb2.get_recommendations(request.form['movie'],cb2.cosine_sim).head(10)
      improved_recommendations_keywords = cb2.improved_recommendations(request.form['movie'], cb2.cosine_sim).head(10)

      recommendations_description.set_index("title", inplace=True)
      recommendations_keywords.set_index("title", inplace=True)
      improved_recommendations_keywords.set_index("title", inplace=True)

      return render_template(
          "content.html",
          recommendations_description = recommendations_description.index.values,
          recommendations_keywords = recommendations_keywords.index.values,
          improved_recommendations_keywords = improved_recommendations_keywords.index.values
      )

@app.route('/collaborative_filtering',methods = ['POST', 'GET'])
def collaborative_filtering():
    performance_svd = []
    performance_nmf = []
    performance_als = []
    performance_sgd = []
    performance_knn = []
    performance_cosine = []
    performance_pearson = []
    if request.method == 'POST':
        if 'predict' in request.form:
            print("Request successful")
            user_id = request.form.get('user_id')
            print(user_id)
            performance_svd = collaborative_filtering_handler.use_svd()
            performance_knn = collaborative_filtering_handler.use_knn()
            performance_als = collaborative_filtering_handler.use_als()
            performance_sgd = collaborative_filtering_handler.use_sgd()
            performance_nmf = collaborative_filtering_handler.use_nmf()
            performance_cosine = collaborative_filtering_handler.use_cosine_similarity()
            performance_pearson = collaborative_filtering_handler.use_pearson_baseline()
            # predictions = collaborative_filtering_handler.make_predictions(user_id)

    return render_template('index.html', performance_svd=performance_svd, performance_knn=performance_knn,
                           performance_als=performance_als, performance_sgd=performance_sgd,
                           performance_nmf=performance_nmf, performance_cosine=performance_cosine,
                           performance_pearson=performance_pearson)


if __name__ == '__main__':
    app.run()
