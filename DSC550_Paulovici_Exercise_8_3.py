#%%[markdown]
# # Week 8: 
# File: DSC550_Paulovici_Exercise_8_3.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 5/2/2020<br>
# Course: DSC 550 Data Mining (2205-1)<br>
# Assignment: 8.3 Exercise: Original Analysis Case Study Part 1, 2, 3

#%%[markdown]
# # Part 1

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
df = pd.read_csv("survey_results_public.csv")
headers = pd.read_csv("survey_results_schema.csv", index_col="Column")

#%%
# Step 2: check the dimension of the table
print("The dimension of the data is: {}".format(df.shape))

#%%
# Step 3 - filter the data
# include only united states respondants 
filt = (df["Country"] == "United States")

df = df.loc[filt]

#%%
# Step 4: check the dimension of the table
print("The dimension of the data is: {}".format(df.shape))

#%%
# Step 5: display the first few rows of the dataset and header file
df.head()

#%%
headers.sort_index(inplace=True)
headers

#%%
# step 6: Update values for YearsCode and YearsCodePro
df["YearsCode"] = df["YearsCode"].replace({"Less than 1 year": 0, "More than 50 years":50})
df["YearsCodePro"] = df["YearsCodePro"].replace({"Less than 1 year": 0, "More than 50 years":50})

#%%
# Step 7: Convert YearsCode and YearsCodePro to numerics
df['Age'] = pd.to_numeric(df['Age'],errors='coerce')
df['YearsCode'] = pd.to_numeric(df['YearsCode'],errors='coerce')
df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'],errors='coerce')

#%%
# step 8: summary data for select features
# age
df["Age"].describe()

#%%
# YearsCode
df["YearsCode"].describe()

#%%
# YearsCode
df["YearsCodePro"].describe()

#%%
# Step 9: Create histograms of Age, YearsCode, and YearsCodePro
# set up the figure size
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 1, ncols = 3)

# Specify the features of interest
num_features = ['Age', 'YearsCode', 'YearsCodePro']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(df[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
plt.show()

#%%
# Step 10: Replace Values for Opensourcer & Employment
# df.rename(columns={"OpenSourcer":"OpenSourcer/year"})
df["OpenSourcer"] = df["OpenSourcer"].replace({"Less than once per year": "< 1 / year", "Once a month or more often": "> 1 / month", "Less than once a month but more than once per year":"< 1 / month"})

df["Employment"] = df["Employment"].replace(
    {"Employed full-time":"Full-Time",
    "Employed part-time": "Part-Time",
    "Independent contractor, freelancer, or self-employed": "Independent",
    "Not employed, and not looking for work":"Not & ot looking",
    "Not employed, but looking for work":"Not & looking"})

#%%
# Step 11: Filter Gender to man and woman only
filt = (df["Gender"] == "Man") | (df["Gender"] == "Woman")
df = df.loc[filt]
df.shape

#%%
# Step 12: Create histograms of OpenSourcer, Gender, Hobbyist, Student, JobSat, MgrIdiot
sns.catplot(y="OpenSourcer", palette="GnBu_d", kind="count", data=df)

#%%
sns.catplot(x="Gender", palette="GnBu_d", kind="count", data=df)
#%%
sns.catplot(x="Hobbyist", palette="GnBu_d", kind="count", data=df)

#%%
sns.catplot(x="Student", palette="GnBu_d", kind="count", data=df)
#%%
sns.catplot(y="MgrIdiot", palette="GnBu_d", kind="count", data=df)

#%%
sns.catplot(y="Employment", palette="GnBu_d", kind="count", data=df)

#%%
# step 13: Scatter/Histogram plots of Salary Vs. Age
sns.scatterplot(y="ConvertedComp", x="Age", palette="GnBu_d", data=df)

#%%
# histogram of Salary
# how many values are below 500K
filt = df["ConvertedComp"] < 500000
df_500 = df.loc[filt]
df_500.ConvertedComp.describe()

#%%
temp = df_500["ConvertedComp"]
plt.hist(temp, bins=40)
plt.show()

#%%[markdown]
# # Part 2

#%%[markdown]
# ## Assignment Tasks
# Create Part 2 of your Analysis Case Study project. Part 2 should consist of Dimensionality and Feature Reduction. You can use any methods/tools you think are most appropriate. <br><br>
# Write the step-by-step instructions for completing the Dimensionality and Feature Reduction part of your case study.

#%%
# Recap of filter done so far (dimensionality reduction)
# Step 3 included only for United States and Python users
# step 11 included only man and woman as genders

#%%
# Step 14: Additional filter

# employement, remove anyone not currently employeed 
filt = (df["Employment"] == "Full-Time") | (df["Employment"] == "Part-Time") | (df["Employment"] == "Independent")
df = df.loc[filt]

#%%
print("The dimension of the data is: {}".format(df.shape))

#%%
# age, only include age range 18 - 65
filt = (df["Age"] > 17) & (df["Age"] < 66)
df = df.loc[filt]

#%%
# ConvertedComp (Salary) 
filt = (df["ConvertedComp"] < 500000)
df = df.loc[filt]
print("The dimension of the data is: {}".format(df.shape))

#%%
# Step 15: Remove features not of interest
# features = ["MainBranch", "Hobbyist", "OpenSourcer", "Employment", "Student",
# "OrgSize","YearsCode","YearsCodePro","CareerSat","JobSat","MgrIdiot",
# "ConvertedComp","WorkWeekHrs","WorkLoc","LanguageWorkedWith","OpSys","BetterLife",
# "Age","Gender","Dependents"]

features = ["MainBranch", "Hobbyist", "Employment", "OpenSourcer", "YearsCode","YearsCodePro","CareerSat",
"ConvertedComp","WorkWeekHrs", "Age","Gender"]

df = df[features]

df.shape

#%%
headers = pd.read_csv("survey_results_schema.csv")

#%%
# filt = (headers["Column"] == "MainBranch") | (headers["Column"] == "Hobbyist") | (headers["Column"] == "OpenSourcer") | (headers["Column"] == "Employment") | (headers["Column"] == "Student") | (headers["Column"] == "OrgSize") | (headers["Column"] == "YearsCode") | (headers["Column"] == "YearsCodePro") | (headers["Column"] == "CareerSat") | (headers["Column"] == "JobSat") | (headers["Column"] == "MgrIdiot") | (headers["Column"] == "ConvertedComp") | (headers["Column"] == "WorkWeekHrs") | (headers["Column"] == "WorkLoc") | (headers["Column"] == "LanguageWorkedWith") | (headers["Column"] == "OpSys") |(headers["Column"] == "BetterLife") | (headers["Column"] == "Age") | (headers["Column"] == "Gender") | (headers["Column"] == "Dependents")

filt = (headers["Column"] == "MainBranch") | (headers["Column"] == "Hobbyist") | (headers["Column"] == "OpenSourcer") | (headers["Column"] == "Employment") | (headers["Column"] == "YearsCode") | (headers["Column"] == "YearsCodePro") | (headers["Column"] == "ConvertedComp") | (headers["Column"] == "WorkWeekHrs") | (headers["Column"] == "Age") | (headers["Column"] == "Gender")

headers = headers.loc[filt]
headers.set_index("Column")
headers.sort_index(inplace=True)
headers

#%%
# step 16: na values
df = df.dropna()
df.shape 

#%%[markdown]
# # Part 3

#%%
# step 17: create target column
# print(df.ConvertedComp)

# avg = 246592.5
avg = df.ConvertedComp.mean()

df["Above_Below_Avg_Sal"] = ""

for index, row in df.iterrows():
    if row.ConvertedComp < avg:
        df.at[index, 'Above_Below_Avg_Sal'] = "Below Avg"
    else:
        df.at[index, 'Above_Below_Avg_Sal'] = "Above Avg"

df.Above_Below_Avg_Sal.describe()

#%%
# step 18: convert categorical data to numbers

# categorical features
cat_features = ['MainBranch','Hobbyist','OpenSourcer','Employment', 'Gender']
df_cat = df[cat_features]

#one hot encoding
df_cat_dummies = pd.get_dummies(df_cat)

print(df_cat_dummies.head())

#%%
# step 19: create whole features dataset for train/validation data splitting

# combine numerical data & dummie features together
features_model = ['YearsCodePro','WorkWeekHrs','Age']
df_model_x = df[features_model]
# df_model_x = pd.concat([df[features_model], df_cat_dummies], axis=1)
# set the target dataset
df_model_y = df['Above_Below_Avg_Sal']

# separate data into training and validation and check the details of the datasets
# import packages
from sklearn.model_selection import train_test_split

# split the data
x_train, x_val, y_train, y_val = train_test_split(df_model_x, df_model_y, test_size =0.3, random_state=11)

# number of samples in each set
print("No. of samples in training set: ", x_train.shape[0])
print("No. of samples in validation set:", x_val.shape[0])

# Above / Below Avg
print('\n')
print('No. of Above / Below Avg in the training set:')
print(y_train.value_counts())

print('\n')
print('No. of Above / Below Avg in the validation set:')
print(y_val.value_counts())

#%%
# step 20: Eval Metrics
from sklearn.linear_model import LogisticRegression

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC

# Instantiate the classification model 
model = LogisticRegression()

#The ConfusionMatrix visualizer taxes a model
classes = ['Below Avg','Above Avg']
cm = ConfusionMatrix(model, classes=classes, percent=False)

# fit the model
cm.fit(x_train, y_train)

#To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
#and then creates the confusion_matrix from scikit learn.
cm.score(x_val, y_val)

# change fontsize of the labels in the figure
for label in cm.ax.texts:
    label.set_size(20)

#How did we do?
cm.poof()

#%%
# Precision, Recall, and F1 Score
# set the size of the figure and the font size 
#%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20

# Instantiate the visualizer
visualizer = ClassificationReport(model, classes=classes)

visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()

#%%
# ROC and AUC
#Instantiate the visualizer
visualizer = ROCAUC(model)

visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()

# %%
