import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# import the training and test data
dftrain = pd.read_csv('train.csv', header=0)
dftest = pd.read_csv('test.csv', header=0)

# drop the columns we will not use
dftrain = dftrain.drop(['Name', 'Ticket', 'Cabin'], axis=1)
dftest = dftest.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# convert string values to dummy variables
dftrain['Sex'] = pd.Categorical(dftrain.Sex).labels
dftest['Sex'] = pd.Categorical(dftest.Sex).labels
dftrain['Embarked'] = pd.Categorical(dftrain.Embarked).labels
dftest['Embarked'] = pd.Categorical(dftest.Embarked).labels

# fill NaN
dftrain.fillna(method='ffill', inplace=True)
dftest.fillna(method='ffill', inplace=True)

# single decision tree
tree = DecisionTreeClassifier()
tree = tree.fit(dftrain.ix[:,'Pclass':], dftrain['Survived'])
dt_output = tree.predict(dftest.ix[:,'Pclass':])
df_dtout = pd.DataFrame(dftest['PassengerId'].values, columns=['PassengerId'])
df_dtout['Survived'] = dt_output

# save decision tree prediction:
df_dtout.to_csv('dt_prediciton.csv', index=False)

# random forest
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(dftrain.ix[:,'Pclass':], dftrain['Survived'])
rf_output = forest.predict(dftest.ix[:,'Pclass':])
df_rfout = pd.DataFrame(dftest['PassengerId'].values, columns=['PassengerId'])
df_rfout['Survived'] = rf_output

# save decision tree prediction:
df_rfout.to_csv('rf_prediciton.csv', index=False)
