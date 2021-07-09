import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load train dataset & drop NaN entries
df = pd.read_csv('../data/train.csv')
df.dropna(inplace=True)

#Q1: What is the most frequently-occurring age for males and females?
age_female = df.loc[df['Sex'] == 'female'].mode()['Age'][0]
age_male = df.loc[df['Sex'] == 'male'].mode()['Age'][0]

print("age_female=", age_female)
print("age_male=", age_male)

#Q2: Convert the 'Sex' feature into a categorical feature called 'Sex_Category
df['Sex_Category'] = df['Sex'].apply(lambda sex: 1 if sex == 'female' else 0)

#Q3: Consider the 'Age' and 'Fare' columns. Bring these two numerical features to a common scale (mean of 0 and stdev of 1).
age_mean = df['Age'].mean()
age_std = df['Age'].std()

fare_mean = df['Fare'].mean()
fare_std = df['Fare'].std()

df['Age_Normalized'] = df['Age'].apply(lambda age: (age - age_mean) / age_std)
df['Fare_Normalized'] = df['Fare'].apply(lambda fare: (fare - fare_mean) / fare_std)

#Q4: Implement the logistic function in Python.
def logistic_fn(x):
    return 1 / (1 + np.exp(-x))

#Q5: Using the newly-generated features Age_Normalized, Fare_Normalized, Sex_Category, implement the Logistic Regression function.
x_train, x_test, y_train, y_test = train_test_split(df[['Age_Normalized', 'Fare_Normalized', 'Sex_Category', 'Pclass']], df[['Survived']], test_size=0.1)
log_reg = LogisticRegression(random_state=2)
log_reg.fit(x_train, y_train)
survival_predictions = log_reg.predict(x_test)
print(classification_report(y_test, survival_predictions))