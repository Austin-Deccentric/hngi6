# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:59:24 2019

@author: Augustine Chukwu
"""

import pandas as pd
import numpy as np

#u_cols = ['id', 'name', 'username', 'email', 'short_bio']
users = pd.read_csv('users.csv',)
print("\nUser Data :")
print("shape : ", users.shape)
print(users.head())

posts = pd.read_csv('posts.csv')
print("\nPost Data :")
print("shape :", posts.shape)
print(posts.head())

following = pd.read_csv('following.csv')
print("\nFollowing Data :")
print("shape :", following.shape)
print(following.head())

notifications = pd.read_csv('notifications.csv')
print("\nNotifications Data :")
print("shape :", notifications.shape)
print(notifications.head())

ratings = pd.read_csv('rating.csv')
print("\nRating Data :")
print("shape :", ratings.shape)
print(ratings.head())


n_users = ratings.user_id.unique().shape[0]
n_posts = ratings.post_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_posts))

for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
    
#Calculating the similiarity between the two features
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosnine')
post_similarity = pairwise_distances(data_matrix.T, metric='cosine')
#the above code gives the post similiarities and user similiarites in an array

#making predictions based on this;we define a funtion
def predict(rating, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis]+ similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)])
    elif type == 'post':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
#making predictions
user_predictions = predict(data_matrix, user_similarity, type='user')
post_prediction = predict(data_matrix, post_similarity, type = 'item')

    