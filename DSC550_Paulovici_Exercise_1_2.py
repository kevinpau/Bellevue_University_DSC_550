#%%[markdown]
# # Week 1: Text Processing, Transformation, Vectorization, and Feature Extraction
# File: DSC550_Paulovici_Exercise_1_2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 3/15/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 1.2 Exercise: Create a Search Engine

#%%[markdown]
# ## Part 1
# Create an array of size [10,50] and use the random function to fill it with integers between 1 and 500 (inclusive). Load it to a DataFrame.

#%%
# load library
import numpy as np
import pandas as pd

# create array
a = np.random.randint(1,500,500)
a = a.reshape(10,50)

# load to Datafram
data = pd.DataFrame(a)
data

#%%[markdown]
# ## Part 2
# Calculate the sum of each row; the sum of each column; and the sum of all entries.

#%%
# a = np.random.randint(1,500,15)
# a = a.reshape(3,5)
# data = pd.DataFrame(a)
# data

#%%
# sum of each row
data.sum(axis=1)

#%%
# sum of each column
data.sum(axis=0)

#%%
# sum of all entries
data.values.sum()

#%%[markdown]
# ## Part 3
# Determine the minimum; maximum; average; and standard deviation of all entries.

#%%
# minimum of all entries
data.values.min()

#%% 
# maximum of all entries
data.values.max()

#%% 
# average of all entries
data.values.mean()

#%%
# standard deviation of all entries
data.values.std()

#%%[markdown]
# ## Part 4
# Sort the DataFrame: by rows, by columns, on row 2, on column 2. <br> <br>
# TODO: This question is confusing to me. I think I'm missing something simple and getting confused by index and col names having the same reference. Can you provide a solution to this?

#%%
# By rows
data.sort_values(list(data.columns),axis=0)

#%%
# By Columns
data.sort_values([0, 1],axis=1)

#%%
# By row 2
data.sort_values(by=2,axis=1)


#%%
# By column 2
data.sort_values(by=2,axis=0)

