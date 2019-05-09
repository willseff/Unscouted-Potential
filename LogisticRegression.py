import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# prepare datasets to be fed in the regression model
dataset = pd.read_csv('Workbook5Clean.csv')
DS_Scorers = pd.read_csv('Group_DataScorers.csv')
DS_Specialists = pd.read_csv('Group_DataSpecialists.csv')
DS_Centers = pd.read_csv('Group_DataCenters.csv')
print(dataset)
#predict attend class given extra hours and grade
CV =  dataset.Drafted.reshape((len(dataset.Drafted), 1))
data = (dataset.ix[:,'Class':'USGp'].values).reshape((len(dataset.Drafted), 32))

# Create a KNN object
LogReg = LogisticRegression()

# Train the model using the training sets
LogReg.fit(data, CV)

# the model
print('Coefficients (m): \n', LogReg.coef_)
print('Intercept (b): \n', LogReg.intercept_)

#predict the class for each data point
predicted = LogReg.predict(data)
print("Predictions: \n", np.array([predicted]).T)

df = pd.DataFrame()
df = np.array([predicted]).T

with open('LRPredictions.csv', 'w') as fp:
   writer = (csv.writer(fp)).writerows(df)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",LogReg.predict_proba(data))

print("Accuracy score for the model: \n", LogReg.score(data,CV))

#print(metrics.confusion_matrix(CV, predicted, labels=["Yes","No"]))

# Calculating 5 fold cross validation results
model = LogisticRegression()
kf = KFold(len(CV), n_folds=10)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 10 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
