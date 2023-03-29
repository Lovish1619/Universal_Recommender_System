from flask import Flask, render_template
import pickle
# from recommender.component.books import BookRecommender
# book_recommend = BookRecommender()
# book_recommend.recommend()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
