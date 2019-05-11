import csv
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Workbook5Clean.csv')

# prepare datasets to be fed into the naive bayes model
CV = dataset.Drafted

data = (dataset.loc[:,'Class':'USGp'].values)

x_train, x_test, y_train, y_test = train_test_split(data,CV,test_size = 0.25 , random_state = 42)

# Create model object
NB = GaussianNB()

# Train the model using the training sets
NB.fit(x_train, y_train)

#Model
print("Probability of the classes: ", NB.class_prior_)
print("Mean of each feature per class:\n", NB.theta_)
print("Variance of each feature per class:\n", NB.sigma_)

#predict the class for each data point
predicted = NB.predict(x_test)
print (predicted)

cm = confusion_matrix(y_test,predicted)

print(cm)

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
