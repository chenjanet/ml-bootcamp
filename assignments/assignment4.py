import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import tree, datasets
from sklearn.metrics import classification_report

train_df = pd.read_csv('../data/titanic/train.csv')

# Feature engineering
train_df[['female', 'male']] = pd.get_dummies(train_df['Sex'])
train_df[['C', 'Q', 'S']] = pd.get_dummies(train_df['Embarked'])
train_df.fillna(method="ffill", inplace=True)
drop_features = ['Sex', 'Ticket', 'Name', 'Cabin', 'Embarked']
train_df.drop(drop_features, inplace=True, axis=1)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(train_df.loc[:, 'Pclass':], train_df.Survived, test_size=0.25, random_state=42)

#Q1: use GridSearchCV to tune hyperparameters and obtain the best model
# Using GridSearchCV on DecisionTreeClassifier
dt_parameter_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]
}

dt_model_gridsearched = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=dt_parameter_grid)
dt_model_gridsearched.fit(x_train, y_train)
dt_gridsearched_predictions = dt_model_gridsearched.predict(x_test)
print('Classification report for DecisionTree:')
print(classification_report(y_test, dt_gridsearched_predictions))

# Using GridSearchCV on RandomForestClassifier
rf_parameter_grid = {
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'n_estimators': [10, 20, 50, 100, 500]
}

rf_model_gridsearched = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=dt_parameter_grid)
rf_model_gridsearched.fit(x_train, y_train)
rf_gridsearched_predictions = rf_model_gridsearched.predict(x_test)
print('Classification report for RandomForest:')
print(classification_report(y_test, rf_gridsearched_predictions))

# Using GridSearchCV on GradientBoostingClassifier
gb_parameter_grid = {
    'loss': ['deviance'],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'n_estimators': [10, 20, 50, 100, 500],
    'criterion': ['mse', 'friedman_mse'],
    'max_depth': [5, 10, 50, 100, 500]
}

gb_model_gridsearched = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=gb_parameter_grid)
gb_model_gridsearched.fit(x_train, y_train)
gb_gridsearched_predictions = gb_model_gridsearched.predict(x_test)
print('Classification report for GradientBoost:')
print(classification_report(y_test, gb_gridsearched_predictions))

# Using GridSearchCV on AdaBoostClassifier
ab_parameter_grid = {
    'n_estimators': [10, 20, 50, 100, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'random_state': [42]
}

ab_model_gridsearched = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=ab_parameter_grid)
ab_model_gridsearched.fit(x_train, y_train)
ab_gridsearched_predictions = ab_model_gridsearched.predict(x_test)
print('Classification report for AdaBoost:')
print(classification_report(y_test, ab_gridsearched_predictions))

# Using GridSearchCV on XGBoostClassifier
xg_parameter_grid = {
    'max_depth': [5, 10, 50, 100, 500],
    'n_estimators': [10, 20, 50, 100, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
}

xg_model_gridsearched = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42), param_grid=xg_parameter_grid)
xg_model_gridsearched.fit(x_train, y_train)
xg_gridsearched_predictions = xg_model_gridsearched.predict(x_test)
print('Classification report for XGBoost:')
print(classification_report(y_test, xg_gridsearched_predictions))

#Q2: Use K-Fold Cross Validation on the dataset
x = train_df.loc[:, 'Pclass':]
y = train_df.Survived
kf = KFold(n_splits=2, random_state=42, shuffle=False)
scores = np.array([])
for train_index, test_index in kf.split(x):
  x_train, x_test = x.iloc[train_index], x.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
  xg_model_gridsearched.fit(x_train, y_train)
  predictions = xg_model_gridsearched.predict(x_test)
  scores = np.append(scores, classification_report(y_test, predictions, output_dict=True)['accuracy'])

print('Final K-fold model performance is ', scores.mean())