import numpy as np
import pandas as pd

# Import train data
train_data = pd.read_csv('train.csv')
train_data.head()
test_data = pd.read_csv('test.csv')
test_data.head()
#print(train_data, '\n')

# Explore a patterns
# 1. Does all the females survided and all the males died?
women = train_data.loc[train_data.Sex == 'female']["Survived"]
#print(women, '\n')
#print(women.shape, '\n')
rate_women = sum(women)/len(women)
#print('# of women survived:', sum(women))
#print('# of all the women:', len(women))
#print('% of women who survived:', rate_women, '\n')

men = train_data.loc[train_data.Sex == 'male']["Survived"]
#print(men, '\n')
#print(men.shape, '\n')
rate_men = sum(men)/len(men)
#print('# of men survived:', sum(men))
#print('# of all the men:', len(men))
#print('% of men who survived:', rate_men, '\n')

# Create the first random forest classifier
from sklearn.ensemble import RandomForestClassifier
y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
# important: we don't have and y_test dataset, so we have to assume that the output is correct. 
# We can't compare it with a standard dataset. To doso, we should split the test.csv in subset train and test dataset