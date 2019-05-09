import csv
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('Workbook5Clean.csv')
print(dataset)

# prepare datasets to be fed into the naive bayes model
#predict attend class given extra hours and grade
CV =  dataset.Drafted.reshape((len(dataset.Drafted), 1))
data = (dataset.ix[:,'Class':'USGp'].values).reshape((len(dataset.Drafted), 32))

# Create model object
NB = GaussianNB()

# Train the model using the training sets
NB.fit(data, CV)

#Model
print("Probability of the classes: ", NB.class_prior_)
print("Mean of each feature per class:\n", NB.theta_)
print("Variance of each feature per class:\n", NB.sigma_)

#predict the class for each data point
predicted = NB.predict(data)
print("Predictions:\n",np.array([predicted]).T)

df = pd.DataFrame()
df = np.array([predicted]).T

with open('NBPredictions.csv', 'w') as fp:
   writer = (csv.writer(fp)).writerows(df)

# predict the probability/likelihood of the prediction
prob_of_pred = NB.predict_proba(data)
print("Probability of each class for the prediction: \n",prob_of_pred)

print("Accuracy of the model: ",NB.score(data,CV))

#print("The confusion matrix:\n", metrics.confusion_matrix(CV, predicted, ['No','Yes']))

# Calculating 5 fold cross validation results
model = GaussianNB()
kf = KFold(len(CV), n_folds=10)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 10 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
