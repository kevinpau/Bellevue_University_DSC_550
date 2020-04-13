#%%[markdown]
# # Week 5: 
# File: DSC550_Paulovici_Exercise_5_2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 4/12/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 5.2 Exercise: Graph Analysis

#%%[markdown]
# ## Assignment Tasks
# Complete the Hypothesis Case Study Part 1 tutorial. It is not a complete case study; it is just the steps you might take to do Graph Analysis. I have provided sample code for you to use as you go through the tutorial. I recommend that you comment out the steps and run them separately so you can fully understand what you are doing for each step of the analysis. As you go through each step, take screenshots to “prove” to me that you successfully completed each step. Paste your screenshots into a Word document and submit that Word document to the Assignment submission link.

# <br><br> Code provided by Prof. Becky Deitenbeck

#%%[markdown]
# ### Case Study:  Testing  Hypothesis
# Hypothesis:  Articles about Climate Change are more likely to be published by "Liberal" sources <br><br>
# NOTE: This case study is not complete!  We are only using the first part of it to practice Graphic Analytics.

#%%
import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from collections import Counter

#%%[markdown]
# #### Step 1:  Load data into a dataframe
#%%
addr1 = "articles1.csv" # file in same dir as .py file
articles = pd.read_csv(addr1)

#%%[markdown]
# #### Step 2:  check the dimension of the table/look at the data

#%%
#Dimension of table
print("The dimension of the table is: {}".format(articles.shape))

#%%
#Display the data
print(articles.head(5))

#%%
#what type of variables are in the table 
print("Describe Data")
print(articles.describe())
print("Summarized Data")
print(articles.describe(include=['O']))

#%%
#display length of data
print(len(articles))
print(len(articles.index)) # another way

#%%
#display publishers (publications)
print(articles.publication.unique())

#%%
#display min, max of years published
print(articles['year'].min())
print(articles['year'].max())

#%%
#display how many articles from each year
print(articles['year'].value_counts())

#%%[markdown]
# #### Step 3:  Create some bar charts to show articles

#%%
#display bar chart of articles sorted by Publication Name
ax = articles['publication'].value_counts().sort_index().plot(kind='bar', fontsize=14, figsize=(12,10))
ax.set_title('Article Publication\n', fontsize=20)
ax.set_xlabel('Publication', fontsize=18)
ax.set_ylabel('Count', fontsize=18);
plt.show()

#%%
#display bar chart of articles sorted by counts
ax = articles['publication'].value_counts().plot(kind='bar', fontsize=14, figsize=(12,10))
ax.set_title('Article Count - most to least\n', fontsize=20)
ax.set_xlabel('Publication', fontsize=18)
ax.set_ylabel('Count', fontsize=18);
plt.show()

#%%[markdown]
# #### Step 4:  clean text:  no punctuation/all lowercase

#%%
def clean_text(article):
    clean1 = re.sub(r'['+string.punctuation + '’—”'+']', "", article.lower())
    return re.sub(r'\W+', ' ', clean1)

articles['tokenized'] = articles['content'].map(lambda x: clean_text(x))
print("clean text:  \n{}".format(articles['tokenized'].head()))

#%%
#look at mean, min, max article lengths
articles['num_wds'] = articles['tokenized'].apply(lambda x: len(x.split()))
print("Mean: {:.2f}".format(articles['num_wds'].mean()))
print("Min:  {:.2f}".format(articles['num_wds'].min()))
print("Max:  {:.2f}".format(articles['num_wds'].max()))

#%%[markdown]
# #### Step 5:  remove articles with no words

#%%
len(articles[articles['num_wds']==0])
articles = articles[articles['num_wds']>0]
print("New Mean:  {:.2f}".format(articles['num_wds'].mean()))
print("New Min:   {:.2f}".format(articles['num_wds'].min()))
print("New Max:   {:.2f}".format(articles['num_wds'].max()))

#%%[markdown]
# #### Step 6:  Check for Outliers:  show bar graph of outliers

#%%
ax=articles['num_wds'].plot(kind='hist', bins=50, fontsize=14, figsize=(12,10))
ax.set_title('Article Length in Words\n', fontsize=20)
ax.set_ylabel('Frequency', fontsize=18)
ax.set_xlabel('Number of Words', fontsize=18);
plt.show()
