import numpy as np
import pandas as pd

# Import train data
train_data = pd.read_csv('train.csv')
train_data.head()
print(train_data, '\n')

# Explore a patterns
# Does all the females survided and all the males died?
women = train_data.loc[train_data.Sex == 'female']["Survived"]
print(women, '\n')
print(women.shape, '\n')
rate_women = sum(women)/len(women)
print('# of women survived:', sum(women))
print('# of all the women:', len(women))
print('% of women who survived:', rate_women, '\n')