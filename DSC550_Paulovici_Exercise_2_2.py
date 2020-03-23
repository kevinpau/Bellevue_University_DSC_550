#%%[markdown]
# # Week 2: 
# File: DSC550_Paulovici_Exercise_2_2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 3/22/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 2.2 Exercise: Build your Text Classifiers

#%%[markdown]
# ## Part 1

# You can find the data for this exercise in the Weekly Resources: Week 2 Data Files.Preparing Text: For this part, you will start by reading the Income.json (controversial-comments.json) file into a DataFrame.

#%%
import json
import pandas as pd
from pandas.io.json import json_normalize

# with open("controversial-comments_subset.json") as fileIn:
# with open("controversial-comments.json") as fileIn:
#     count = 0

#     # read each line as a json obj
#     for line in fileIn:
#         line = line.strip()
#         data = json.loads(line)

#         # create the dataframe for the first obj else append data
#         if count == 0:
#             df = json_normalize(data)            
#         else:
#             df2 = json_normalize(data)
#             df = df.append(df2, ignore_index=True)
        
#         count += 1

with open("controversial-comments.json") as f:
    df = pd.DataFrame(json.loads(line) for line in f)   

df.head(10)        

#%%[markdown]
# ### Part A
# Convert all text to lowercase letters.

#%%
df["txt"] = df["txt"].str.lower()
df.head(5)


#%%[markdown]
# ### Part B
# Remove all punctuation from the text. 

#%%
import unicodedata
import sys

punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

df.txt = df.txt.apply(lambda x: x.translate(punctuation))
df.head(5)

#%%[markdown]
# ### Part C
# Remove stop words.

#%%
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

# load stop words
stop_words = stopwords.words('english')

# TODO: get lambda function working
# remove stop_words
# df.txt.apply(lambda x: [item for item in x if item not in stop_words])

# remove stop words
def stopWords(sentence):
    tokens = sentence.split()
    w = [item for item in tokens if item not in stop_words]
    return ' '.join(w)

df.txt = df.txt.apply(stopWords)
df.head(5)

#%%[markdown]
# ### Part D
# Apply NLTK’s PorterStemmer. 

#%%
from nltk.stem.porter import PorterStemmer

# creater stemmer
porter = PorterStemmer()

# TODO: get lambda function working
# df.txt.apply(lambda x: [porter.stem(word) for word in x])

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df.txt = df.txt.apply(stem_sentences)
df.head(5)

#%%[markdown]
# ### Part E
# Use a Tf-idf vector instead of the word frequency vector.

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = df.txt
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

print(feature_matrix)
print(feature_matrix.toarray())
print(tfidf.vocabulary_)

#%%
#%%[markdown]
# ## Part 2
# Complete the 5.3 Encoding Dictionaries of Features examples. Be sure to read the Discussion concerning keeping track of how many times a word is used in a document. Also be sure to run the example and read the Discussion from 6.9 Weighting Word Importance. Finally, consider tokenizing words or sentences (see 6.4) and tagging parts of speech (see 6.7) Be sure to review how to encode days of the week (see 7.6). <br><br>

# You can start with the #1 program and add to it or you can start a new program.

#%% [markdown]
# ### 5.3 Encoding Dictionaries of Features - book examples

#%%
from sklearn.feature_extraction import DictVectorizer

# create dict
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

# create dict vectorizer
dictvectorizer = DictVectorizer(sparse=False)

# convert dict to feature matrix
features = dictvectorizer.fit_transform(data_dict)

# view feature matrix
features

#%%
# get feature names
feature_names = dictvectorizer.get_feature_names()

# view feature names
feature_names

#%%
# create dataframe to output data better
import pandas as pd

# create df from features
pd.DataFrame(features, columns=feature_names)

#%%
# create word counts dictionaries for four documents
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# create list 
doc_word_counts = [doc_1_word_count, doc_2_word_count, doc_3_word_count, doc_4_word_count]

# convert list of word count dicts into feature matrix
dictvectorizer.fit_transform(doc_word_counts)


#%% [markdown]
# ### 6.4 Tokenizing Text - book examples

#%%
from nltk.tokenize import word_tokenize

# create text
string = "The science of today is the technology of tomorrow"

# tokenize words
word_tokenize(string) # error

# work around for viewing only
# string.split()

#%%
# Tokenize sentences
from nltk.tokenize import sent_tokenize

# create text
string = "The science of today is the technology of tomorrow. Tomrow is today."

# tokenize sentences
sent_tokenize(string) # error

# work around for viewing only
# string.split(".")

#%% [markdown]
# ### 6.7 Tagging Parts of Speech - book examples

#%%
from nltk import pos_tag
from nltk import word_tokenize

# create text
text_data = "Chris loved outdoor running"

# use pre-trained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))

# show parts of speech
text_tagged

[word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

#%%
import nltk
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

# create text
tweets = ["I am eating a burrito for breakfast", 
"Political science is an amazing field", 
"San Francisco is an awesome city"]

# Create list
tagged_tweets = []

# tag each word and each tweet
for tweet in tweets:
    tweet_tag = nltk.pos_tag(nltk.word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# use one-hot encoding to conver the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)

#%%
one_hot_multi.classes_

#%% [markdown]
# ### 6.9  - Weighting Word Importance book examples

#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# create_text 
text_data = np.array(["I love Brazil. Brazil!", 
"Sweden is best", "Germany beats both"])

# create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# show tf-idf feature matrix
feature_matrix

#%%
# show tf-idf featyre matrix as dense matrix
feature_matrix.toarray()

#%%
# show feature names
tfidf.vocabulary_


#%% [markdown]
# ### 7.6  Encoding Days of the week - book examples

#%%
import pandas as pd

# create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# show days of the week
dates.dt.weekday_name

# %%
# show days of the week
dates.dt.weekday

#%%[markdown]
# ### Part A
# Provide me with an example (besides counting words in a document) of how these techniques could be used. (Just a couple sentences.)

# We can use the words from some source to help determine a few things besides just word count. For example, if we consider an article, we can determine the subject of the article by how often a word appears. If economy or economics appears often we can say the subject or point or the article is about economics. Similiarlly, we can determine the mood or gender by looking for key words (e.g., good, great, she, him, etc).

#%%[markdown]
# ### Part C
# You can create a datafile file or use one from the course resources. You must use DataFrames!

# %%
# create a dataframe for Part B.
import pandas as pd

data = [["Person_1", "I hate the outdoors!"],
        ["Person_2", "The outdoors is okay."],
        ["Person_3", "I can't stand to be inside! Outside is my home."]]

df = pd.DataFrame(data, columns = ['Name', "Statement"])

df

#%%[markdown]
# ### Part B
# Then implement at least 3 of these Text techniques in a program demonstrating how your example could be accomplished. Be sure to include lots of comments.

#%%
from nltk.tokenize import word_tokenize

# mood and subject are what we are looking to determine from the df

# create list of words for mood
good = ["good", "happy", "love"]
bad = ["bad", "hate", "sad"]

# tokenize statements of each row to compare to list for mood (example 6.4)
statements = df.Statement

# if the tokenized word matches our mood list they get a point
# the higher points determins the mood of the statement
for s in statements:
    good_mood = 0
    bad_mood = 0

    wt = word_tokenize(s)

    for word in wt:
        if word in good:
            good_mood +=1
        
        elif word in bad:
            bad_mood +=1
    
    if good_mood > bad_mood:
        print(s, "This persons statement indicates a good mood about the subject.\n")
    elif bad_mood > good_mood:
        print(s, "This persons statement indicates a bad mood about the subject.\n")
    else:
        print(s, "This persons statement didn't indicate a good or bad mood about the subject.\n")

# %%
# since we also want to try to predict the subject, lets wigh the words of the statements. (example 6.9)
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(statements)
print(feature_matrix, "\n")
print(feature_matrix.toarray(),"\n")
pprint(tfidf.vocabulary_)

# I initially thought that the subject, outdoors, would be the most important. However, since it is frequent within the document it becomes less important. On the otherhand, the mood (e.g., hate, okay) become the most important word of each persons statement.

# %%
# determining parts of speach to see if the statement is about the person (example 6.7)
from nltk import pos_tag
from nltk import word_tokenize

for s in statements:
    text_tagged = pos_tag(word_tokenize(s))

    # print personal pronouns
    personal = [word for word, tag in text_tagged if tag in ["PRP", "PRP$"]]
    print(personal, "\n")

# %%
