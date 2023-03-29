import ast
import os
import sys
import pickle

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from recommender.exception import CustomException
from recommender.logger import logging


class PopularTransformer:
    """
    This class is responsible for creating the dataframe for the popular recommendations.
    args:
    rating_file_path: Path of rating file
    books_file_path: Path of book file
    """
    logging.info('We are in Popular Transformer Class')

    def __init__(self, movies_file_path, credit_file_path):
        try:
            self.movies_file = os.path.basename(movies_file_path)
            self.credit_file = os.path.basename(credit_file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def merge_ratings_and_name(self):
        """
        This function is responsible for merging the rating and books file.
        return: Combined dataframe of rating and books
        """
        logging.info('We are in merge function of Popular Transformer Class')
        try:
            rating_file = self.rating_file
            books_file = self.books_file
            logging.info('Loading the rating and books file into dataframe.')
            rating_df = pd.read_csv(rating_file)
            books_df = pd.read_csv(books_file)
            logging.info('Files loaded successfully')
            books_ratings_df = rating_df.merge(books_df, on='ISBN')
            logging.info('Dataframes merged successfully')
            return books_ratings_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def book_votes(self):
        """
        This function is responsible for counting the number of votes for a book in a dataframe.
        return: Dataframe with book votes
        """
        logging.info('We are in book_votes function of PopularTransformer Class')
        try:
            books_ratings_df = self.merge_ratings_and_name()
            vote_df = books_ratings_df.groupby('Book-Title').count()['Book-Rating'].reset_index()
            vote_df.rename(columns={'Book-Rating': 'Book-Votes'}, inplace=True)
            logging.info("Function executed successfully")
            return vote_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def average_rating(self):
        """
        This function is responsible for calculating the average votes for each book in a dataframe.
        return: Dataframe with average rating
        """
        logging.info("We are in average_rating function.")
        try:
            books_ratings_df = self.merge_ratings_and_name()
            rating_df = books_ratings_df.groupby('Book-Title').mean()['Book-Rating'].reset_index()
            rating_df.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)
            logging.info('Function executed successfully.')
            return rating_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def popular_data_frame(self):
        """
        This function is responsible for merging the dataframes having book votes and average rating.
        return: Dataframe having book details with votes and average rating
        """
        logging.info('We are in popular_data_frame function.')
        try:
            vote_df = self.book_votes()
            rating_df = self.average_rating()
            popular_df = vote_df.merge(rating_df, on='Book-Title')
            logging.info('Function executed successfully.')
            return popular_df
        except Exception as e:
            raise CustomException(e, sys) from e


class ContentTransformer:
    """
    This class is responsible for creating the dataframe through which content based filtering can be done.
    args:
    rating_file_path: Path of rating file
    books_file_path: Path of book file
    """
    logging.info('We are in ContentTransformer class.')

    def __init__(self, movies_file_path, credits_file_path):
        try:
            self.movies_file = os.path.basename(movies_file_path)
            self.credits_file = os.path.basename(credits_file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def merge_movies_and_credits(self):
        """
        This function is responsible for merging the movies and credits file.
        return: Combined dataframe of movies and credits
        """
        logging.info('We are in merge function of Content Transformer Class')
        try:
            movies_file = self.movies_file
            credits_file = self.credits_file
            logging.info('Loading the movies and credits file into dataframe.')
            movies_df = pd.read_csv(movies_file)
            credits_df = pd.read_csv(credits_file)
            logging.info('Dataframes Loaded Successfully.')
            movies_credits_df = movies_df.merge(credits_df, on='title')
            logging.info('Dataframes merged successfully.')
            movies_credits_df = movies_credits_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            return movies_credits_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def handling_null(self):
        try:
            movies_credits_df = self.merge_movies_and_credits()
            movies_credits_df.dropna(inplace=True)
            return movies_credits_df
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def convert_dict_to_list(self, object):
        try:
            L = []
            for i in ast.literal_eval(object):
                L.append(i["name"])
            return L
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def fetch_cast(self, object):
        try:
            L = []
            counter = 0
            for i in ast.literal_eval(object):
                if counter!=3:
                    L.append(i['name'])
                    counter += 1
                else:
                    break
            return L
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def fetch_director(self, object):
        try:
            L = []
            for i in ast.literal_eval(object):
                if i["job"] == "Director":
                    L.append(i['name'])
                    break
            return L
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def text_stemmer(self, text):
        try:
            ps = PorterStemmer()
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)
        except Exception as e:
            raise CustomException(e, sys) from e

    def tag_generator(self):
        try:
            movies_credits_df = self.handling_null()
            movies_credits_df['genres'] = movies_credits_df['genres'].apply(self.convert_dict_to_list())
            movies_credits_df['genres'] = movies_credits_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_credits_df['keywords'] = movies_credits_df['keywords'].apply(self.convert_dict_to_list())
            movies_credits_df['keywords'] = movies_credits_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_credits_df['cast'] = movies_credits_df['cast'].apply(self.fetch_cast())
            movies_credits_df['cast'] = movies_credits_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_credits_df['crew'] = movies_credits_df['crew'].apply(self.fetch_director())
            movies_credits_df['crew'] = movies_credits_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_credits_df['overview'] = movies_credits_df['overview'].apply(lambda x: x.split())
            movies_credits_df['tags'] = movies_credits_df['overview'] + movies_credits_df['genres'] + movies_credits_df['keywords'] + movies_credits_df['cast'] + movies_credits_df[
                'crew']
            movies_credits_df['tags'] = movies_credits_df['tags'].apply(lambda x: " ".join(x))
            movies_credits_df = movies_credits_df[['movie_id, title, tags']]
            movies_credits_df['tags'] = movies_credits_df['tags'].apply(lambda x: x.lower())
            movies_credits_df['tags'] = movies_credits_df['tags'].apply(self.text_stemmer())
            return movies_credits_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def valuable_users(self):
        """
        This function is responsible for finding the experienced users who gave rating for more than 200 books.
        return: Dataframe of valuable users
        """
        logging.info('We are in valuable_users function of CollaborativeTransformer Class')
        try:
            book_rating_df = self.merge_ratings_and_name()
            valuable_users = book_rating_df.groupby('User-ID').count()['Book-Rating'] > 200
            valuable_users = valuable_users[valuable_users].index
            logging.info('Function executed successfully')
            return valuable_users
        except Exception as e:
            raise CustomException(e, sys) from e

    def book_user_df(self):
        """
        This function is responsible for finding the best books with more than 50 votes and merge them with users
        return: Merged dataframe of books and users
        """
        logging.info('We are in book_user_df of CollaborativeTransformer class.')
        try:
            book_rating_df = self.merge_ratings_and_name()
            valuable_users = self.valuable_users()
            filtered_rating = book_rating_df[book_rating_df['User-ID'].isin(valuable_users)]
            famous_books = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
            famous_books = famous_books[famous_books].index
            logging.info('Books and users matching relevant criteria are selected.')
            book_user_df = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
            logging.info('Function executed successfully.')
            return book_user_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def collab_df(self):
        """
        This function is responsible for making the pivot of books and user
        return: Dataframe of pivot of user and books
        """
        logging.info('We are in collab_df function of CollaborativeTransformer class.')
        try:
            book_user_df = self.book_user_df()
            collab_df = book_user_df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
            collab_df.fillna(0, inplace=True)
            logging.info('Function executed successfully.')
            return collab_df
        except Exception as e:
            raise CustomException(e, sys) from e


class BookRecommender:
    """
    This class is responsible for recommendations of both popular and collaborative type.
    args:
    popular_df: Dataframe for popular filtering
    collab_df: Dataframe for collaborative filtering
    books_file: Path of book file
    """
    logging.info('We are in BookRecommender class')

    def __init__(self, popular_df, collab_df, books_file):
        try:
            self.popular_df = popular_df
            self.collab_df = collab_df
            self.books_file = books_file
        except Exception as e:
            raise CustomException(e, sys) from e

    def popular_recommender(self):
        """
        This function is responsible to find the top 50 popular books
        return: Dataframe containing top 50 popular books
        """
        logging.info('We are in popular_recommender function of BookRecommender class.')
        try:
            popular_df = self.popular_df[self.popular_df['Book-Votes'] >= 250].sort_values('Average-Rating',
                                                                                           ascending=False).head(50)
            popular_df = popular_df.merge(self.books_file, on='Book-Title').drop_duplicates('Book-Title')[
                ['Book-Title', 'Book-Author', 'Image-URL-M', 'Book-Votes', 'Average-Rating']]
            return popular_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def similarity_score(self):
        """
        This function is responsible to find the similarity score for all the books.
        """
        logging.info('We are in similarity_score function of BookRecommender Class')
        try:
            similarity_score = cosine_similarity(self.collab_df)
            logging.info('Similarity score calculated successfully.')
            return similarity_score
        except Exception as e:
            raise CustomException(e, sys) from e

    def recommend(self, book_name):
        """
        This function is responsible to recommend the suggestions
        param:
        book_name: Name of book for which recommendations are required
        return: The dataframe for recommendations
        """
        try:
            popular_book = pickle.load(open('popular_book.pkl', 'rb'))
            similarity = pickle.load(open('movie_similarity.pkl', 'rb'))
            book_index = np.where(self.collab_df.index == book_name)[0][0]
            similar_items = sorted(list(enumerate(similarity[book_index])), key=lambda x: x[1], reverse=True)[1:6]
            return similar_items, popular_book
        except Exception as e:
            raise CustomException(e, sys) from e
