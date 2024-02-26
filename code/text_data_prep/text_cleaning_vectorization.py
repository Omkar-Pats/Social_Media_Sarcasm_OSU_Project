'''
Code for Cleaning Text Data and Vectorizing the Text Data
Takes about 5 mins to completely execute
'''

##########################################################################################
###################################### Imports ###########################################

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk downloads required
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



##########################################################################################
###################################### Load Data #########################################

df = pd.read_csv('data/base_data/base-data-sarcasm.csv')



##########################################################################################
############################### Basic Exploration ########################################

print("Columns: ", df.columns, end="\n\n")

print("Initial Shape: ", df.shape, end="\n\n")
print(df.isna().sum(), end="\n\n")

# just 7 comments are na - we can drop these 7 rows. we have a lot of data
df.dropna(axis=0, inplace=True)
print("Shape After Dropping NA: ", df.shape, end="\n\n")



##########################################################################################
############### Segregate the main Text and other Non-Text Columns #######################

# comments - keeping id for joining back later
df_comments_data = df[['id', 'comment']].copy()
print(df_comments_data.head(3), end="\n\n")

# parent comments - keeping id for joining back later
df_parent_comment_data = df[['id', 'parent_comment']].copy()
print(df_parent_comment_data.head(3), end="\n\n")

# remaining columns
df_remaining_data = df[df.columns.difference(['comment', 'parent_comment'])].copy()
print(df_remaining_data.head(3), end="\n\n")



##########################################################################################################
#################### Main Class for Performing the Text Cleaning and Vectorization #######################

class TextDataPrep:
    '''
    Helper class for performing text dataprep
    '''
    
    def __init__(self, input_df:pd.DataFrame, text_column_name:str='comment'):
        '''
        Inputs:
            input_df: pandas dataframe containing a text column
            text_column_name: name of the text column
        '''
        self.df = input_df
        self.text_column_name = text_column_name
        
        #setup stopwords to be used
        # 'not' is an exception because it will be useful in sentiment analysis - sarcasm is a type of sentiment analysis
        self.eng_stopwords = stopwords.words('english')
        self.eng_stopwords = [i for i in self.eng_stopwords if i!='not']
            
    def remove_stopwords(self, txt):
        '''
        helper method for removing stopwords
        '''
        txt = txt.split(' ')

        txt = [i for i in txt if i not in self.eng_stopwords]

        final_txt = ' '.join(txt)

        return final_txt
    
    @staticmethod
    def _tokezine_lemmatize(txt):
        '''
        helper method for tokenizing and lemmatizing
        '''
        tokenized_txt = word_tokenize(txt)

        wordnet_lemmatizer = WordNetLemmatizer()

        lemmatized_txt = [wordnet_lemmatizer.lemmatize(w) for w in tokenized_txt]

        final_txt = ' '.join(lemmatized_txt)

        return final_txt
            
    
    def run_data_prep_pipeline(self):
        '''
        method for performing the data prep for text data
        '''
        
        # convert to lowercase
        self.df['cleaned_'+self.text_column_name] = self.df[self.text_column_name].str.lower()
        
        # remove numeric values - keep text only
        self.df['cleaned_'+self.text_column_name] = self.df['cleaned_'+self.text_column_name].str.replace("[^a-zA-Z']", " ", regex=True)
        
        # remove stopwords
        self.df['cleaned_'+self.text_column_name] = self.df['cleaned_'+self.text_column_name].apply(self.remove_stopwords)
        
        # perform lemmatization
        self.df['cleaned_'+self.text_column_name] = self.df['cleaned_'+self.text_column_name].apply(TextDataPrep._tokezine_lemmatize)
        
        return self.df
    
    
    def perform_tfidf_vectorization(self):
        '''
        method for performing tfidf vectorization
        
        Returns: 
            a high dimensional pandas dataframe with tfidf values
        '''
        tfidf = TfidfVectorizer(
            ngram_range=(1,3),
            max_features=70,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            min_df=0.0,
            max_df=0.80
        )
        
        comments_tfidf = tfidf.fit_transform(
            self.df['cleaned_'+self.text_column_name]
        )
        
        feature_names = tfidf.get_feature_names_out()
        
        comments_tfidf_df = pd.DataFrame(
            comments_tfidf.toarray(), 
            columns = feature_names
        )

        comments_tfidf_df = pd.concat(
            [self.df, comments_tfidf_df],
            axis = 1
        ).reset_index(drop=True)
        
        print(comments_tfidf_df.info(), end="\n\n")
        
        return comments_tfidf_df



#######################################################################################
############################# Apply on "comment" column ###############################

comments_data_prep = TextDataPrep(
    input_df = df_comments_data,
    text_column_name = 'comment'
)
# perform data prep and cleaning first
df_comments_data_clean = comments_data_prep.run_data_prep_pipeline()

# peform tfidf vectorization after cleaning
df_comments_data_tfidf = comments_data_prep.perform_tfidf_vectorization()

print(df_comments_data_tfidf.head(3), end="\n\n")

df_comments_data_tfidf.to_csv('data/vectorized_text_data/comments_data_vectorized.csv', index=False)



#######################################################################################
########################## Apply on "parent_comment" column ###########################

parent_comments_data_prep = TextDataPrep(
    input_df = df_parent_comment_data,
    text_column_name = 'parent_comment'
)
# perform data prep and cleaning first
df_parent_comment_data_clean = parent_comments_data_prep.run_data_prep_pipeline()

# peform tfidf vectorization after cleaning
df_parent_comment_data_tfidf = parent_comments_data_prep.perform_tfidf_vectorization()

print(df_parent_comment_data_tfidf.head(3), end="\n\n")

df_parent_comment_data_tfidf.to_csv('data/vectorized_text_data/parent_comments_data_vectorized.csv', index=False)
