#%%[markdown]
# # Week 4: 
# File: DSC550_Paulovici_Exercise_4_2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 4/5/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 4.2 Exercise: Calculate Document Similarity

#%%[markdown]
# ## Assignment Tasks
# Create a scenario of when and why you might want to determine if comments are positive or negative (or male/female or pass/fail or any other “binary” categorization). Also tell me how the results could be used. <br><br>
# You must read the data in from a file.<br><br>
# You must use some kind of vectorization method/tool (my example uses sklearn count.vectorizer but you can use any vectorization tool or Jaccard Distance.<br><br>
# Create some kind of a dictionary of sample words you will use to search /categorize your data.<br><br>
# Display the results.<br><br>
# For 10% extra credit…add something more to your program that relates to Ch 5-7!

#%%[markdown]
# ### Scenario
# For this assignment I will use movie comments to determine if the viewers have a positive or negative view of the movie. Movie reviews are common and widely used on certain sites. Having highly positive reviews can act as a marketing tool and help influence potential watchers to go spend their money on it.

#%%
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

#%%
# read in data and create dataframe
data = 'movie_responses.csv'
df = pd.read_csv(data)
print(df)

# create corpus from the responses
corpus = df['Response']

# %%
# create vector of words from responses
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)
print("The vectorized words are: \n{}".format(vectorizer.get_feature_names()))
print("Feature words are: \n{}".format(x.toarray()))

#%%
# create new dataframe and check for positive/negative reviews
reviews = pd.DataFrame({"response": corpus.str.lower()}) # use lower case 
print(reviews, "\n\n")

# list of positive/negative words
pos_list = ["good", "excellent", "great", "enjoy"]
neg_list = ["bad", "boring", "waste", "terrible"]

# check for positive/negative words
def check_reviews(pos_neg, words):
    """
    @pos_neg (string) - positive or negative 
    @words (list) - list of words to check for
    """ 
    # create an empty column for pos/neg first
    reviews[pos_neg] = 0

    for index, row in reviews.iterrows():
        for word in words:
            if word in row.response:
                reviews.at[index, pos_neg] = reviews.at[index, pos_neg] + 1

check_reviews("positive", pos_list)
check_reviews("negative", neg_list)

# check total
reviews["Total"] = reviews.positive - reviews.negative

print(reviews)

#%%
s = sum(reviews["Total"])
print("\nOverall Score: {}\n".format(s))

if s > 0:
    print("Overall reviews are good")
elif s < 0:
    print("Overall reviews are negative")
else:
    print("Overall reviews are neutral")
# %%


