# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Aquire and show  data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]
#print(train_df.columns.values)
print('Train dataset info:')
train_df.info()
print('-'*40)
print('Test dataset info:')
test_df.info()
print('-'*40)
print('Train dataset descibed:', '\n', train_df.describe())
print('-'*40)
print('Train dataset descibed:', '\n', train_df.describe(include='O'))
print('-'*40)

# Analyze by pivoting features
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*40)
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*40)
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*40)
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*40)

# Analyze by visualizing data
gAge = sns.FacetGrid(train_df, col='Survived')
gAge.map(plt.hist, 'Age', bins=20)
plt.show()

gClass = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)
gClass.map(plt.hist, 'Age', alpha=.5, bins=20)
gClass.add_legend()
plt.show()

gEmbarked = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
gEmbarked.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
gEmbarked.add_legend()
plt.show()

# Wrangle data

# Model Predict and solve


