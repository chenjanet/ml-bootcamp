import pandas as pd
import re

test_df = pd.read_csv('./test.csv')
train_df = pd.read_csv('./train.csv')

#Q1: Create a list of integers starting from 1 to 100,000 where each number is squared.
squared_integers = [ x**2 for x in range(1, 100001)]

#Q2: Create a function to print the maximum of two given values.
max_two_numbers = lambda x, y: x if x > y else y

#Q3: How many columns and rows are there in the test data?
row, col = test_df.shape
print("row = ", row)
print("col = ", col)

#Q4: What are the youngest and oldest ages recorded on the ship in the train data?
age_youngest = train_df['Age'].describe()['min']
age_oldest = train_df['Age'].describe()['max']
print("age_youngest = ", age_youngest)
print("age_oldest = ", age_oldest)

#Q5: What are the proportions of male and female passengers in the train data?
female_count = sum(train_df['Sex'] == 'female')
male_count = sum(train_df['Sex'] == 'male')
print("female_count = ", female_count)
print("male_count = ", male_count)

#Q6: What is the average passenger ticket price per class in the train data?
p_1_avg_fare = round(train_df.loc[train_df['Pclass'] == 1, 'Fare'].sum() / sum(train_df['Pclass'] == 1), 2)
p_2_avg_fare = round(train_df.loc[train_df['Pclass'] == 2, 'Fare'].sum() / sum(train_df['Pclass'] == 2), 2)
p_3_avg_fare = round(train_df.loc[train_df['Pclass'] == 3, 'Fare'].sum() / sum(train_df['Pclass'] == 3), 2)
print("p_1_avg_fare = ", p_1_avg_fare)
print("p_2_avg_fare = ", p_2_avg_fare)
print("p_3_avg_fare = ", p_3_avg_fare)

#Q7: How many different titles are present in the passenger names in the train data?
title_count = {}
for name in train_df['Name']:
    title = re.findall(" [A-Za-z]+\.", name)[0][1:-1]
    if not title in title_count:
        title_count[title] = 1
    else:
        title_count[title] += 1

print("title_count = ", title_count)


