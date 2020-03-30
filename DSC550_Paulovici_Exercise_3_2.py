#%%[markdown]
# # Week 3: 
# File: DSC550_Paulovici_Exercise_3_2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 3/29/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 3.2 Exercise: Graph Analysis

#%%[markdown]
# ## Assignment Tasks
# You can create a new analysis scenario or you can use the tutorials you completed this week.<br><br>
# A) Display the same analysis using 3 different charts (ex. A bar chart, a line chart and a pie chart)<br>
# B) Use appropriate, complete, professional labeling.<br>
# C) Rank your charts from most effective to least effective.<br>
# D) Write a 300-word paper justifying your ranking.

#%%[markdown]
# ### Data
# I'll be working with Corona Virus data for NY counties. Data was retrieved from https://coronavirus.health.ny.gov/county-county-breakdown-positive-cases for daily updates.

#%%
# Start by importing the csv file into a DataFrame
import pandas as pd

data = pd.read_csv("corona_virus_NY2.csv")

# preview data
data.head(5)

# %%
# I'll be using the latest data row (date) so I'll make a subset of data to graph
latest_data = data.filter(["County", "3/29/2020"])
latest_data = latest_data.sort_values("3/29/2020", ascending=False)
latest_data

#%%
# We don't need to plot all the counties, it would be too chaotic, so lets select the top 5
top_5 = latest_data.loc[latest_data["3/29/2020"] > 2000]
top_5

#%%
import matplotlib.pyplot as plt
import numpy as np

# %%[markdown]
# ### Bar chart
#%%
# get countries list
counties = top_5["County"].values

# get values of countries list
val = top_5["3/29/2020"].values

plt.barh(counties, val)
plt.ylabel("County")
plt.xlabel("Confirmed Cases")
plt.title("Corona Virus Cases in NY")
plt.show()

# %%[markdown]
# ### Line chart
#%%
plt.plot(counties, val)
plt.scatter(counties, val)
plt.xlabel("County")
plt.ylabel("Confirmed Cases")
plt.title("Corona Virus Cases in NY")
plt.show()

# %%[markdown]
# ### Pie chart
#%%
explode = (0.1, 0, 0, 0, 0)
plt.pie(val, explode=explode, labels=counties, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Corona Virus Cases in NY")
plt.show()

#%%[markdown]
# ### Summary
# Parts A and B are completed as shown above. <br><br>
# For Part C, I would rank these from most effective to least effective as the following: bar chart, pie chart, and the line/scatter. <br><br>
# For Part D - see summary below. <br><br>
# Visualization is a key component of the data science process. Being able to effectively display analytical results in in a comprehendible way to is a skill and can take some trial-and-error until the right format of the plot is selected. The details, such as color and labels can also play an important role in the process. For this assignment, a bar chart, a scatter/line plot, and a pie chart are used to demonstrate this. <br><br>
# Before discussing how I ranked these various plots, it is important to cover what data was used and what is the intent of it. I chose to use confirmed corona cases by counties in NY. Additionally, for plotting purpose I down selected to the most recent data and only focused on the top 5 counties. For my subset of data, I wanted to demonstrate the vast different seen by these counties; this heavily influenced how I ranked these plots. However, some consideration was given if additional counties were included.
# I ranked the bar chart as the most effective, followed by the pie chart, and the scatter/line plot I found to be least effective. The horizontal bar was most effective to me because it provided a clear and easy comparison with a rough estimation of confirmed cases. While the pie chart does not directly provided the count of cases, it gives a percentage view comparison to the other counties. Again, this is clear and direct. However, by expanding the counties, the pie chart would quickly lose its effectiveness because of too much data, even with the inclusion of a legend it would be chaotic to pin point the counties. In that scenario I would rank the pie chart last. The scatter/line chart was ranked last because I did not think it was as effective as the bar chart. However it does clearly label all points and give an estimation of cases. If this plot was not sorted it could easily become a chaotic by increasing and decreasing cases. Expanding the counties would extend the plot horizontal, being able to scroll vertical (bar chart) is more effective. <br><br>
# This exercised demonstrated multiple plots can be used for a given set of data. However, they are not equally effective at demonstrating the intended purposes. Using a variety of plots to fit the data is an import task for a data scientist to reach the intended audience.  
