#%%[markdown]
# # Week 6: 
# File: DSC550_Paulovici_Exercise_6_3.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 4/19/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 6.3 Exercise: Original Analysis Case Study Part 1

#%%[markdown]
# ## Assignment Tasks
# Provide a short narrative describing an original idea for an analysis problem. Find or create appropriate data that can be analyzed. <br>
# Write the step-by-step instructions for completing the Graph Analysis part of your case study.

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick

#%%
# Step 1: load the data into a dataframe
file_name = "vgsales.csv" 
data = pd.read_csv(file_name)

#%%
# Step 2: check the dimension of the table
print("The dimension of the data is: {}".format(data.shape))

#%%
# Step 3: Clean up data - only include Year 2014 - present
data = data.loc[data["Year"] > 2013]

#%%
# Step 4: check the dimension of the table
print("The dimension of the new data is: {}".format(data.shape))

#%%
# Step 5: look at the data
print(data.head(5))

#%%
# Step 6: describe data
print("Describe Data\n")
print(data.describe())
print("\nSummarized Data\n")
print(data.describe(include=['O']))

#%% 
# Step 7: histogram of genre/platform 
sns.set(style="darkgrid")
# platform
sns.countplot(x="Platform", palette="GnBu_d", data=data)

#%%
# After reviewing the game counts for the platform, we can elminate some of the poorer performers and the older systems.
data = data.loc[(data["Platform"] == "PS4") | (data["Platform"] == "3DS") | (data["Platform"] == "XOne") | (data["Platform"] == "WiiU") | (data["Platform"] == "PC") | (data["Platform"] == "PSV")]

sns.countplot(x="Platform", palette="GnBu_d", data=data)

#%%
# genre
sns.countplot(x="Genre", palette="GnBu_d", data=data)

#%%
# After reviewing the game counts for the genre, we can elminate some of the poorer performers since we want to reach a wide auidence.
data = data.loc[(data["Genre"] == "Shooter") | (data["Genre"] == "Action") | (data["Genre"] == "Sports") | (data["Genre"] == "Adventure")]

sns.countplot(x="Genre", palette="GnBu_d", data=data)

#%%
# Step 8: scatter plots of sales vs. genre/platform

# platform
sns.catplot(y="Platform", x="Global_Sales", kind="swarm", data=data)


# %%
# genre
sns.catplot(y="Genre", x="Global_Sales", kind="swarm", data=data)

# %%
# sns.catplot(x="Genre", y="Global_Sales", col="Platform", aspect=.6, kind="swarm", data=data)

sns.catplot(x="Platform", y="Global_Sales", col="Genre", col_wrap=2, aspect=.6, kind="swarm", data=data)


# %%
