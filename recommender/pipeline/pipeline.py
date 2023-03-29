import pickle
import sys, os
from recommender.logger import logging
from recommender.exception import CustomException
from recommender.component import books


def books_recommendation_pipeline():
    try:
        popular_obj = books.PopularTransformer('D:\Study\Data Science\Book_Recommender_System\Data\Ratings.csv'
                                               , 'D:\Study\Data Science\Book_Recommender_System\Data\Books.csv')
        popular_df = popular_obj.popular_data_frame()
        collab_obj = books.CollaborativeTransformer('D:\Study\Data Science\Book_Recommender_System\Data\Ratings.csv'
                                                    , 'D:\Study\Data Science\Book_Recommender_System\Data\Books.csv')
        collab_df = collab_obj.collab_df()
        recommend = books.BookRecommender(popular_df, collab_df,
                                          'D:\Study\Data Science\Book_Recommender_System\Data\Books.csv')
        popular_books = recommend.popular_recommender()
        popular_books.to_pickle('popular_book.pkl')
        similarity_score = recommend.similarity_score()
        similarity_score.to_pickle('movie_similarity.pkl')
    except Exception as e:
        raise CustomException(e, sys) from e


def movies_recommendation_pipeline():
    try:
        pass
    except Exception as e:
        raise CustomException(e, sys) from e


def fashion_recommendation_pipeline():
    try:
        pass
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == '__main__':
    choice = input("Choose the pipeline for train: ")
    if choice == 'books':
        books_recommendation_pipeline()
    if choice == 'movies':
        movies_recommendation_pipeline()
    if choice == 'fashion':
        fashion_recommendation_pipeline()
    else:
        print("Enter a valid choice.")
