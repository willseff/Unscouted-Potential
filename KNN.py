import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import pickle

dataset = pd.read_csv('Workbook5Clean.csv')
print(dataset)

# prepare datasets to be fed in the regression model
#predict attend class given extra hours and grade
CV =  dataset.Drafted.reshape((len(dataset.Drafted), 1))
data = (dataset.ix[:,'Class':'USGp'].values).reshape((len(dataset.Drafted), 32))

# Create a KNN object
KNN = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
KNN.fit(data, CV)

#predict the class for each data point
predicted = KNN.predict(data)
print("Predictions: \n", np.array([predicted]).T)

df = pd.DataFrame()
df = np.array([predicted]).T

with open('KNNPrediction.csv', 'w') as fp:
   writer = (csv.writer(fp)).writerows(df)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",KNN.predict_proba(data))

print("Neighbors and their Distance: \n",KNN.kneighbors(data, return_distance=True))

print("Accuracy score for the model: \n", KNN.score(data,CV))

#metrics.confusion_matrix(CV, predicted, labels=["Yes","No"]))

# Calculating 5 fold cross validation results
model = KNeighborsClassifier()
kf = KFold(len(CV), n_folds=10)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 10 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))