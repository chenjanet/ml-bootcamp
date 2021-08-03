import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

train_df = pd.read_csv('../data/titanic/train.csv')
test_df = pd.read_csv('../data/titanic/test.csv')

#Q1: What is the average ticket price per ticket class for each individual port of embarkation?
avg_ticket_price = train_df.groupby(by=['Pclass', 'Embarked']).mean()['Fare'].to_dict()
print('avg_ticket_price = ', avg_ticket_price)

#Q2: Create a new feature called 'Title' in the dataset.
train_df['Title'] = train_df['Name'].apply(lambda name: re.findall('[A-Za-z]+\.', name)[0])
test_df['Title'] = test_df['Name'].apply(lambda name: re.findall('[A-Za-z]+\.', name)[0])

#Q3: Encode the new 'Title' feature using get_dummies()
train_df = pd.concat([train_df, pd.get_dummies(train_df['Title'])], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['Title'])], axis=1)

#Q4: Create a Logistic Regression model with the data. Train your model on the data in train.csv, and then make predictions on test.csv.
train_df[['female', 'male']] = pd.get_dummies(train_df['Sex'])
train_df[['C', 'Q', 'S']] = pd.get_dummies(train_df['Embarked'])
train_df.fillna(method='ffill', inplace=True)

test_df[['female', 'male']] = pd.get_dummies(test_df['Sex'])
test_df[['C', 'Q', 'S']] = pd.get_dummies(test_df['Embarked'])
test_df.fillna(method='ffill', inplace=True)

drop_features = ['PassengerId', 'Sex', 'Ticket', 'Name', 'Cabin', 'Embarked', 'Title']
train_df.drop(drop_features, inplace=True, axis=1)
test_df.drop(drop_features, inplace=True, axis=1)

x_train, x_test, y_train, y_test = train_test_split(train_df.loc[:, 'Pclass':], train_df.Survived, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)

predictions = model.predict(test_df.loc[:,'Pclass':])
print(predictions)
