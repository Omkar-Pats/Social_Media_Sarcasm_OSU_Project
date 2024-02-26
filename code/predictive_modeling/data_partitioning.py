'''
Code for splitting the data into training, validation and test sets. 

This process will be carried out for two different types of datasets 
(pca transformed and untransformed) and we will have a total of 6 datasets
'''


##########################################################################################
###################################### Imports ###########################################

import pandas as pd
from sklearn.model_selection import train_test_split


################################################################################################
###################################### Load Datasets ###########################################

df = pd.read_csv('data/final_data/final_master_data.csv')

print(df.head(3), end="\n\n")
print("-------------"*10, end="\n\n")

vec_df_comments = pd.read_csv('data/vectorized_text_data/comments_data_vectorized.csv')
vec_df_p_comments = pd.read_csv('data/vectorized_text_data/parent_comments_data_vectorized.csv')

# load the topic sentiment data extracted from the sentiment analysis
topic_sentiment_data = pd.read_csv('data/topic_sentiment_data/topic_sentiment_data.csv')

# merge the topic sentiment analysis data to the original dataset
df = pd.merge(
    df, 
    topic_sentiment_data,
    on = 'id',
    how = 'left'
)
df['age_of_account'] = (pd.Timestamp.today() - pd.to_datetime(df['date'])).dt.days


print(vec_df_comments.head(3), end="\n\n")
print("-------------"*10, end="\n\n")

print(vec_df_p_comments.head(3), end="\n\n")
print("-------------"*10, end="\n\n")


##################################################################################################
####################### Basic Cleaning before Splitting the Datasets #############################

# rename a few columns to make it more understandable - comments data
new_col_name = []
for col in vec_df_comments.columns:
    if col not in ['id', 'comment', 'cleaned_comment']:
        new_col_name.append(col + '_vec_comments')
    else:
        new_col_name.append(col)
        
vec_df_comments.columns = new_col_name

print(vec_df_comments.head(3))
print("-------------"*10, end="\n\n")


# rename a few columns to make it more understandable - parent comments data
new_col_name = []
for col in vec_df_p_comments.columns:
    if col not in ['id', 'parent_comment', 'cleaned_parent_comment']:
        new_col_name.append(col + '_vec_p_comments')
    else:
        new_col_name.append(col)
        
vec_df_p_comments.columns = new_col_name

print(vec_df_p_comments.head(3))
print("-------------"*10, end="\n\n")


# merge the unvectorized data to create a consolidated dataset
merged_vec_comments = pd.merge(
    vec_df_comments,
    vec_df_p_comments, 
    on = 'id',
    how = 'inner'
)


# drop any NAs before splitting the dataset
df = df.dropna()
merged_vec_comments = merged_vec_comments.dropna()


final_df_vectorized = pd.merge(
    df[
        ['id', 'cleaned_comment', 'comment', 'cleaned_parent_comment', 'parent_comment',
         'label', 'author', 'subreddit', 'score', 'ups', 'downs', 'date', 'fear', 'trust', 'disgust', 'surprise', 'topic', 'age_of_account']
    ],
    merged_vec_comments[['id'] + list(merged_vec_comments.columns.difference(df.columns))],
    on = 'id'
)


# print shape of the datasets before splitting 
print("Shape of PCA Transformed Dataset: ", df.shape)
print("-------------"*10, end="\n\n")

print("Shape of Untransformed Vectorized Dataset: ", final_df_vectorized.shape)
print("-------------"*10, end="\n\n")



##########################################################################################################################
####################### Split the PCA Transformed Dataset into Train(60%)-Val(20%)-Test(20%) #############################

_X_train, X_test, _Y_train, Y_test = train_test_split(
    df[df.columns.difference(['label'])],
    df[['id', 'label']],
    test_size=0.20,
    stratify=df[['label', 'subreddit']],
    random_state=42
)


X_train, X_val, Y_train, Y_val = train_test_split(
    _X_train,
    _Y_train,
    test_size=0.25,
    stratify=pd.concat(
        [
            _X_train['subreddit'].reset_index(drop=True),
            _Y_train['label'].reset_index(drop=True)
        ],
        axis=1
    ),
    random_state=42
)


print("~~~~~~~~~ Splitting the PCA Transformed Data ~~~~~~~~~", end="\n\n")

print("Training Predictors Shape: ", X_train.shape)
print("Training Target Shape: ", Y_train.shape, end="\n\n")

print("Validation Predictors Shape: ", X_val.shape)
print("Validation Target Shape: ", Y_val.shape, end="\n\n")

print("Test Predictors Shape: ", X_test.shape)
print("Test Target Shape: ", Y_test.shape, end="\n\n")

print("-------------"*10, end="\n\n")


# save as csv files
X_train.to_csv('data/predictive_modeling_data/X_train_transformed.csv', index=False)
X_val.to_csv('data/predictive_modeling_data/X_val_transformed.csv', index=False)
X_test.to_csv('data/predictive_modeling_data/X_test_transformed.csv', index=False)

Y_train.to_csv('data/predictive_modeling_data/Y_train_transformed.csv', index=False)
Y_val.to_csv('data/predictive_modeling_data/Y_val_transformed.csv', index=False)
Y_test.to_csv('data/predictive_modeling_data/Y_test_transformed.csv', index=False)



########################################################################################################################
####################### Split the UnTransformed Dataset into Train(60%)-Val(20%)-Test(20%) #############################


_X_train2, X_test2, _Y_train2, Y_test2 = train_test_split(
    final_df_vectorized[final_df_vectorized.columns.difference(['label'])],
    final_df_vectorized[['id', 'label']],
    test_size=0.20,
    stratify=final_df_vectorized[['label', 'subreddit']],
    random_state=42
)


X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
    _X_train2,
    _Y_train2,
    test_size=0.25,
    stratify=pd.concat(
        [
            _X_train2['subreddit'].reset_index(drop=True),
            _Y_train2['label'].reset_index(drop=True)
        ],
        axis=1
    ),
    random_state=42
)


print("~~~~~~~~~ Splitting the Non-Transformed Vectorized Data ~~~~~~~~~", end="\n\n")

print("Training Predictors Shape: ", X_train2.shape)
print("Training Target Shape: ", Y_train2.shape, end="\n\n")

print("Validation Predictors Shape: ", X_val2.shape)
print("Validation Target Shape: ", Y_val2.shape, end="\n\n")

print("Test Predictors Shape: ", X_test2.shape)
print("Test Target Shape: ", Y_test2.shape, end="\n\n")

print("-------------"*10, end="\n\n")

# save as csv files
X_train2.to_csv('data/predictive_modeling_data/X_train_untransformed.csv', index=False)
X_val2.to_csv('data/predictive_modeling_data/X_val_untransformed.csv', index=False)
X_test2.to_csv('data/predictive_modeling_data/X_test_untransformed.csv', index=False)

Y_train2.to_csv('data/predictive_modeling_data/Y_train_untransformed.csv', index=False)
Y_val2.to_csv('data/predictive_modeling_data/Y_val_untransformed.csv', index=False)
Y_test2.to_csv('data/predictive_modeling_data/Y_test_untransformed.csv', index=False)
