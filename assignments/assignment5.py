#Import the necessary libraries here
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('../data/housing/train.csv')

# Q1: Write a function to print the average marks for a given student's name, given two inputs to the function: a dictionary with the grades, and the name of the student.
def student_avg_score(score_dictionary, student_name):
    return round(sum(score_dictionary[student_name]) / len(score_dictionary[student_name]), 2)

# Q2: Find the number of missing values for each column in your data
df_missing_values = train_df.isnull().sum(axis=0).to_dict()
print('df_missing_values = ', df_missing_values)

# Q3: Find the number of two-story houses in the dataset that have an overall quality greater than or equal to 5
good_quality_houses = len(train_df[(train_df['OverallQual'] >= 5) & (train_df['HouseStyle'] == '2Story')])

# Q4: Find the number of duplex houses that have an overall quality less than 5
bad_quality_houses = len(train_df[(train_df['OverallQual'] < 5) & (train_df['BldgType'] == 'Duplex')])

# Q5: Use the matplotlib library to plot a histogram showing the distribution of the following data:
# Overall quality of the houses:
overall_qual_dist = plt.hist(x='OverallQual', data=train_df)

# Roof style of the houses:
roof_style_dist = plt.hist(x='RoofStyle', data=train_df)