#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from IPython.display import display
import numpy as np


############################# Load PCA Transformed Datasets #############################

df_reduced_comment = pd.read_csv('data/pca_data/df_comments_PCA_20.csv')
df_reduced_p_comment = pd.read_csv('data/pca_data/df_p_comments_PCA_15.csv')


display(df_reduced_comment)
display(df_reduced_p_comment)


############################# Merge the Two PCA Transformed Datasets #############################

df_combined = pd.merge(
    df_reduced_comment,
    df_reduced_p_comment,
    how='inner',
    on = 'id'
)
print(df_combined.head())

print(df_combined.columns)


############################# Merge the above combined dataset with the Base Dataset #############################

df_base = pd.read_csv('data/base_data/base-data-sarcasm.csv')
print(df_base.head())


df_base_1 = df_base[['label','id', 'author', 'subreddit', 'score', 'ups', 'downs', 'date']].copy()
print(df_base_1.head())



df_final = pd.merge(
    df_combined,
    df_base_1,
    on='id',
    how='inner'
)

print(df_final.head())
print(df_final.columns)


############################# Merge the user data with the merged dataset created above #############################

user_data = pd.read_csv('data/user_data/user_info.csv')

df_final_v2 = pd.merge(
    df_final,
    user_data, 
    on = 'author', 
    how = 'inner'
)

print(df_final_v2.head())
print(df_final_v2.columns)



############################# Log Transform Skewed Variables before Creating Final Dataset #############################

# log transformation for score
df_final_v2['log_score'] = (
    ((df_final_v2['score'])/np.abs(df_final_v2['score'])).fillna(1)
) * np.log(np.abs(df_final_v2['score']) + 0.001)


# log transformation for ups
df_final_v2['log_ups'] = (
    ((df_final_v2['ups'])/np.abs(df_final_v2['ups'])).fillna(1)
) * np.log(np.abs(df_final_v2['ups']) + 0.001)


# log transformation for comment karma
df_final_v2['log_comment_karma'] = (
    ((df_final_v2['comment_karma'])/np.abs(df_final_v2['comment_karma'])).fillna(1)
) * np.log(np.abs(df_final_v2['comment_karma']) + 0.001)


# log transformation for post karma
df_final_v2['log_post_karma'] = (
    ((df_final_v2['post_karma'])/np.abs(df_final_v2['post_karma'])).fillna(1)
) * np.log(np.abs(df_final_v2['post_karma']) + 0.001)


print(df_final_v2.isnull().sum())
print(df_final_v2.head())
print(df_final_v2.dtypes)


############################# Save the Final Dataset as CSV #############################

df_final.to_csv('data/final_data/final_master_data.csv',index=False)
