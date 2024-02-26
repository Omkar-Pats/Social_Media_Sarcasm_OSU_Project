###############################
# - Calling necessary libraries
###############################

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist
from nltk.chunk import conlltags2tree, tree2conlltags

from nrclex import NRCLex
from collections import Counter
from wordcloud import WordCloud

# Set figure size
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams["figure.autolayout"] = True


# Import text data
os.chdir(r'C:\Users\roedj\Documents\GitHub\homework_MSIS5223\project-deliverable-2-cia')

reddit_data = pd.read_csv(r'data\final_data\final_master_data.csv')

###############################
# Sentiment Analysis
# - Trust vs. Fear
###############################

## Trust ##
###########

# Create an empty list to store "trust" words
trust_words = []

# Iterate over each comment and extract the "trust" words
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        for word in text_object.affect_dict.keys():
            if 'trust' in text_object.affect_dict[word]:
                trust_words.append(word)

# Create a Pandas Series to count the frequency of each "trust" word
trust_word_counts = pd.Series(trust_words).value_counts()

# Plot a bar chart of the most frequent "trust" words
trust_word_counts.head(20).plot(kind='bar')
plt.title('Most Frequent "Trust" Words in Reddit Comments')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Get the top 20 most frequent words
top_words = Counter(trust_words).most_common(20)

# Create a dictionary of word frequencies
word_freqs = {word: freq for word, freq in top_words}

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freqs)

# Display the word cloud image
plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud of Top 20 "Trust" Words in Reddit Comments')
plt.axis('off')
plt.show()

## Fear ##
###########

# Create an empty list to store "fear" words
fear_words = []

# Iterate over each comment and extract the "fear" words
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        for word in text_object.affect_dict.keys():
            if 'fear' in text_object.affect_dict[word]:
                fear_words.append(word)

# Create a Pandas Series to count the frequency of each "fear" word
fear_word_counts = pd.Series(fear_words).value_counts()

# Plot a bar chart of the most frequent "fear" words
fear_word_counts.head(20).plot(kind='bar')
plt.title('Most Frequent "Fear" Words in Reddit Comments')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Get the top 20 most frequent words
top_words = Counter(fear_words).most_common(20)

# Create a dictionary of word frequencies
word_freqs = {word: freq for word, freq in top_words}

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freqs)

# Display the word cloud image
plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud of Top 20 "Fear" Words in Reddit Comments')
plt.axis('off')
plt.show()


## Trust vs. Fear ##
####################

# Apply NRC Emotion Lexicon to each comment and calculate the affect frequency of fear and trust words
freqs = []
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        freqs.append({'fear': text_object.affect_frequencies['fear'],
                      'trust': text_object.affect_frequencies['trust']})
    else:
        freqs.append({'fear': 0,
                      'trust': 0})
freqs_df = pd.DataFrame(freqs)


# Plot the frequency of fear and trust words
ax = round(freqs_df.mean(),4).plot(kind='bar')
plt.title('Average Affect Frequency of Fear and Trust Words in a Reddit Comment')
plt.xlabel('Emotions')
plt.ylabel('Average Affect Frequency')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12)
plt.show()

# Combine the fear and trust frequencies with the sarcasm column
reddit_text_with_freqs = pd.concat([reddit_data, freqs_df], axis=1)
reddit_text_with_freqs = reddit_text_with_freqs[['id','comment','cleaned_comment', 'fear', 'trust', 'label']]

# Calculate the mean fear and trust frequencies for each sarcasm group
mean_freqs = reddit_text_with_freqs[['fear', 'trust', 'label']].groupby('label').mean()

# Plot Average Affect Frequency of Sarcasm and No Sarcasm for Trust and Fear
mean_freqs = mean_freqs.sort_index(ascending=False)
colors = {'fear': 'wheat', 'trust': 'cornflowerblue'}
ax = round(mean_freqs,4).plot(kind='bar', stacked=False, color=[colors[l] for l in mean_freqs.columns])
plt.title('Average Affect Frequency of Fear and Trust Words by Sarcasm Label in a Reddit Comment')
plt.xlabel('Sarcasm Label')
plt.ylabel('Average Affect Frequency')
plt.xticks([0,1], ['No Sarcasm','Sarcasm'],rotation=0)
plt.legend(['Fear', 'Trust'], loc='upper center')

# Add data labels to the plot
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12)

plt.show()

# Check result against data
# with sarcasm
extraction_1 = reddit_text_with_freqs[((reddit_text_with_freqs['fear'] != 0) | 
                                    (reddit_text_with_freqs['trust'] != 0)) &
                                    (reddit_text_with_freqs['label'] == 1)].reset_index(drop=True)
# Without sarcasm
extraction_0 = reddit_text_with_freqs[((reddit_text_with_freqs['fear'] != 0) | 
                                    (reddit_text_with_freqs['trust'] != 0)) &
                                    (reddit_text_with_freqs['label'] == 0)].reset_index(drop=True)

# randomly pick indices
random_indices = []
for i in range(2):
    random_index = random.randint(0, len(extraction_1) - 1)
    random_indices.append(random_index)
    
# print selected reddit comments
for i in random_indices:
    print(f'''
Reddit Comment: {extraction_1['comment'][i]}
Fear Weight: {extraction_1['fear'][i]}
Trust Weight: {extraction_1['trust'][i]}
Sarcasm: {extraction_1['label'][i]}
''')
    
# Repeat for no sarcasm
random_indices = []
for i in range(2):
    random_index = random.randint(0, len(extraction_0) - 1)
    random_indices.append(random_index)
    
for i in random_indices:
    print(f'''
Reddit Comment: {extraction_0['comment'][i]}
Fear Weight: {extraction_0['fear'][i]}
Trust Weight: {extraction_0['trust'][i]}
Sarcasm: {extraction_0['label'][i]}
''')

###############################
# Sentiment Analysis
# - Surprise vs. Disgust
###############################

## Surprise ##
##############

# Create an empty list to store "surprise" words
surprise_words = []

# Iterate over each comment and extract the "surprise" words
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        for word in text_object.affect_dict.keys():
            if 'surprise' in text_object.affect_dict[word]:
                surprise_words.append(word)

# Create a Pandas Series to count the frequency of each "surprise" word
surprise_word_counts = pd.Series(surprise_words).value_counts()

# Plot a bar chart of the most frequent "surprise" words
surprise_word_counts.head(20).plot(kind='bar')
plt.title('Most Frequent "Surprise" Words in Reddit Comments')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Get the top 20 most frequent words
top_words = Counter(surprise_words).most_common(20)

# Create a dictionary of word frequencies
word_freqs = {word: freq for word, freq in top_words}

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freqs)

# Display the word cloud image
plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud of Top 20 "Surprise" Words in Reddit Comments')
plt.axis('off')
plt.show()

## Disgust ##
#############

# Create an empty list to store "disgust" words
disgust_words = []

# Iterate over each comment and extract the "disgust" words
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        for word in text_object.affect_dict.keys():
            if 'disgust' in text_object.affect_dict[word]:
                disgust_words.append(word)

# Create a Pandas Series to count the frequency of each "disgust" word
disgust_word_counts = pd.Series(disgust_words).value_counts()

# Plot a bar chart of the most frequent "disgust" words
disgust_word_counts.head(20).plot(kind='bar')
plt.title('Most Frequent "Disgust" Words in Reddit Comments')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Get the top 20 most frequent words
top_words = Counter(disgust_words).most_common(20)

# Create a dictionary of word frequencies
word_freqs = {word: freq for word, freq in top_words}

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freqs)

# Display the word cloud image
plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud of Top 20 "Disgust" Words in Reddit Comments')
plt.axis('off')
plt.show()

## Disgust vs. Surprise ##
##########################

# Apply NRC Emotion Lexicon to each comment and calculate the affect frequency of disgust and surprise words
freqs = []
for comment in reddit_data['cleaned_comment']:
    if not pd.isna(comment) and isinstance(comment, str):
        text_object = NRCLex(comment)
        freqs.append({'disgust': text_object.affect_frequencies['disgust'],
                      'surprise': text_object.affect_frequencies['surprise']})
    else:
        freqs.append({'disgust': 0,
                      'surprise': 0})
freqs_df = pd.DataFrame(freqs)


# Plot the frequency of disgust and surprise words
ax = round(freqs_df.mean(),4).plot(kind='bar')
plt.title('Average Affect Frequency of Disgust and Surprise Words in a Reddit Comment')
plt.xlabel('Emotions')
plt.ylabel('Average Affect Frequency')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12)
plt.show()

# Combine the disgust and surprise frequencies with the sarcasm column
reddit_text_with_freqs = pd.concat([reddit_text_with_freqs, freqs_df], axis=1)

# Calculate the mean fear and trust frequencies for each sarcasm group
mean_freqs = reddit_text_with_freqs[['disgust', 'surprise', 'label']].groupby('label').mean()
mean_freqs = mean_freqs.sort_index(ascending=False)

# Plot Average Affect Frequency of Sarcasm and No Sarcasm for Disgust and Surprise
colors = {'disgust': 'wheat', 'surprise': 'cornflowerblue'}
ax = round(mean_freqs,4).plot(kind='bar', stacked=False, color=[colors[l] for l in mean_freqs.columns])
plt.title('Average Affect Frequency of Sarcasm and No Sarcasm for Disgust and Surprise')
plt.xlabel('Sarcasm Label')
plt.ylabel('Average Affect Frequency')
plt.xticks([0,1], ['No Sarcasm', 'Sarcasm'], rotation=0)
plt.legend(['Disgust', 'Surprise'], loc='upper center')

# Add data labels to the plot
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12)

plt.show()

# Check result against data
# with sarcasm
extraction_1 = reddit_text_with_freqs[((reddit_text_with_freqs['disgust'] != 0) | 
                                    (reddit_text_with_freqs['surprise'] != 0)) &
                                    (reddit_text_with_freqs['label'] == 1)].reset_index(drop=True)
# without sarcasm
extraction_0 = reddit_text_with_freqs[((reddit_text_with_freqs['disgust'] != 0) | 
                                    (reddit_text_with_freqs['surprise'] != 0)) &
                                    (reddit_text_with_freqs['label'] == 0)].reset_index(drop=True)

# randomly pick indices
random_indices = []
for i in range(2):
    random_index = random.randint(0, len(extraction_1) - 1)
    random_indices.append(random_index)
    
# print selected reddit comments
for i in random_indices:
    print(f'''
Reddit Comment: {extraction_1['comment'][i]}
Disgust Weight: {extraction_1['disgust'][i]}
Surprise Weight: {extraction_1['surprise'][i]}
Sarcasm: {extraction_1['label'][i]}
''')

# Repeat for no sarcasm
random_indices = []
for i in range(2):
    random_index = random.randint(0, len(extraction_0) - 1)
    random_indices.append(random_index)
    
# print selected tweets
for i in random_indices:
    print(f'''
Reddit Comment: {extraction_0['comment'][i]}
Disgust Weight: {extraction_0['disgust'][i]}
Surprise Weight: {extraction_0['surprise'][i]}
Sarcasm: {extraction_0['label'][i]}
''')

###############################
# Topic Analysis
###############################

# Using Latent Dirichlet allocation (LDA) create 4 topics. (1 pt.)
vectorizer = CountVectorizer(max_df=0.8, min_df=10, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(reddit_text_with_freqs['cleaned_comment'].values.astype('U'))

LDA = LatentDirichletAllocation(n_components=4, random_state=35)
LDA.fit(doc_term_matrix)

# What do each of the topics represent? Justify your answer in Word. (1 pt.)
# Printing out top 10 words for each topic
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

# Add a new column to your dataframe containing the LDA topic number. (1 pt.)

topic_values = LDA.transform(doc_term_matrix)

topic_values.shape

reddit_text_with_freqs['topic'] = topic_values.argmax(axis=1)

print(reddit_text_with_freqs.groupby('topic').mean()[['fear','trust','disgust','surprise','label']])

# Export topic and sentiment data for modeling
reddit_text_with_freqs[['id','fear','trust','disgust','surprise','topic']].to_csv(r'data/topic_sentiment_data/topic_sentiment_data.csv', index=False)

# Overall validation
# randomly pick indices
random_indices = []
for i in range(5):
    random_index = random.randint(0, len(reddit_text_with_freqs) - 1)
    random_indices.append(random_index)
    
# print selected reddit comments
for i in random_indices:
    print(f'''
Reddit Comment: {reddit_text_with_freqs['comment'][i]}
Fear Weight: {reddit_text_with_freqs['fear'][i]}
Trust Weight: {reddit_text_with_freqs['trust'][i]}
Disgust Weight: {reddit_text_with_freqs['disgust'][i]}
Surprise Weight: {reddit_text_with_freqs['surprise'][i]}
Topic: {reddit_text_with_freqs['topic'][i]}
Sarcasm: {reddit_text_with_freqs['label'][i]}
''')

###############################
# Named-Entity Recognition
###############################

# Create a function to extract PERSON, ORGANIZATION and LOCATION entities from data
def comment_ner(chunker):
    treestruct = ne_chunk(pos_tag(word_tokenize(chunker)))
    entityp = []
    entityo = []
    entityl = []

    for x in str(treestruct).split('\n'):
        if 'PERSON' in x:
            entityp.append(x)
        elif 'ORGANIZATION' in x:
            entityo.append(x)
        elif 'GPE' in x or 'GSP' in x:
            entityl.append(x)
    return entityp, entityo, entityl

# Function to plot most frequent names for each entities and their association with sarcasm

def plot_entity_top_freq_with_sarcasm(comments, entity):
    comments_fulllist = []
    sarcasm_fulllist = []

    for i in range(len(comments)):
        comments_fulllist.extend(comments[entity][i])
        for x in comments[entity][i]:
            sarcasm_fulllist.append(comments['label'][i])

    # extract clean names from entity using replace
    comments_fulllist = [x.replace(f'({entity}', '').replace('/NNP)', '').replace('/NNP','').replace('/NN','').replace('(GPE', '').replace('/JJ', '').replace('(', '').replace(')', '').strip() for x in comments_fulllist]

    # combine comments_fulllist and sarcasm_fulllist into a list of tuples
    pairs = list(zip(comments_fulllist, sarcasm_fulllist))

    # count the frequency of each pair
    pair_counts = Counter(pairs)

    # create a DataFrame from the counter
    df = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Count'])

    # split the Pair column into Name and SarcasmLabel columns
    df[['Name', 'SarcasmLabel']] = pd.DataFrame(df['Pair'].tolist())

    # pivot the DataFrame to get the desired format
    result = df.pivot(index='Name', columns='SarcasmLabel', values='Count').fillna(0).reset_index().rename_axis(None, axis=1)
    result['Name'] = result['Name'].str.capitalize()

    # plot the frequency chart
    result = result.rename(columns={0:"0",1:"1"})
    result['TotFreq'] = result['0'] + result['1']
    top_names = result.sort_values('TotFreq', ascending=False).head(10).drop('TotFreq', axis=1).set_index('Name')
    colors = {'1': 'salmon', '0': 'lightgreen'}
    ax = top_names.plot(kind='bar', stacked=True, color=[colors[l] for l in result.sort_values('TotFreq', ascending=False).head(10).drop('TotFreq', axis=1).set_index('Name').columns])
    plt.title(f'Top 10 Most Common Names for {entity}')
    plt.xlabel('Name')
    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.legend(['No Sarcasm', 'Sarcasm'])
    plt.show()

# Name extraction and store in new column
reddit_text_with_freqs['PERSON'] = ''
reddit_text_with_freqs['ORGANIZATION'] = ''
reddit_text_with_freqs['LOCATION'] = ''

i = 0
for x in reddit_text_with_freqs['comment'].fillna(''):
    entityp, entityo, entityl = comment_ner(x)
    reddit_text_with_freqs.at[i,'PERSON'] = entityp
    reddit_text_with_freqs.at[i,'ORGANIZATION'] = entityo
    reddit_text_with_freqs.at[i,'LOCATION'] = entityl
    i += 1

# Plot every entity
plot_entity_top_freq_with_sarcasm(reddit_text_with_freqs, 'PERSON')
plot_entity_top_freq_with_sarcasm(reddit_text_with_freqs, 'LOCATION')
plot_entity_top_freq_with_sarcasm(reddit_text_with_freqs ,'ORGANIZATION')