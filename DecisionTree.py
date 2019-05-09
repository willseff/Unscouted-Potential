import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score

# prepare datasets to be fed in the regression model
dataset = pd.read_csv('Workbook5Clean.csv')

DS_Scorers = pd.read_csv('Group_DataScorers.csv')
DS_Specialists = pd.read_csv('Group_DataSpecialists.csv')
DS_Centers = pd.read_csv('Group_DataCenters.csv')
#predict attend class given extra hours and grade
CV =  dataset.Drafted.reshape((len(dataset.Drafted), 1))
data = (dataset.ix[:,'Class':'USGp'].values).reshape((len(dataset.Drafted), 32))

#print(dataset)
#print(data)
#print(CV)

# Create linear regression object
DT = DecisionTreeClassifier(criterion="entropy", min_samples_leaf = 2)

# Train the model using the training sets
DT.fit(data, CV)

tree.export_graphviz(DT, out_file='tree.dot')

#predict the class for each data point
predicted = DT.predict(data)

print("Predictions: \n", np.array([predicted]).T)

df = pd.DataFrame()
df = np.array([predicted]).T

with open('DTPredictions_TEST.csv', 'w') as fp:
   writer = (csv.writer(fp)).writerows(df)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",DT.predict_proba(data))

print("Feature importance: ", DT.feature_importances_)

print("Accuracy score for the model: \n", DT.score(data,CV))

# Calculating 5 fold cross validation results
model = DecisionTreeClassifier()
kf = KFold(len(CV), n_folds=10)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 10 fold cross validation: ", abs(scores))
print("Mean of the 10 fold cross-validation: %0.2f" % abs(scores.mean()))