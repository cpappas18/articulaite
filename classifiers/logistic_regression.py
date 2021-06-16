import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from smote import smote_binary, smote_multiclass # smote.py in the repo

from sklearn.utils.testing import ignore_warnings # to remove warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def logreg(df, n=1, smote='False'):

  # remove first column if indexes
  if df.columns[[0]] == 'Unnamed: 0':
    df.drop(df.columns[[0]], axis=1, inplace=True) 

  # initializing variables
  X = df.iloc[:, n:]
  y = df.iloc[:, :n]
  y_pred = pd.DataFrame()
  modelsDict = {}

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # over/undersampling (n=100) on train set only
  if smote==True:
    if n>1:
      training = pd.concat([y_train, X_train], axis=1)
      training = training.reset_index().iloc[:, 1:]
      training = smote_multiclass(training, {0: 100, 1: 100, 2:100, 3:100}, one_hot_encoded=True)
    else:
      training = pd.concat([y_train, X_train], axis=1)
      training = smote_binary(training)
    X_train = training.iloc[:, n:]
    y_train = training.iloc[:, :n]

  for i in range(n):

    # fitting the model 
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train.iloc[:, i])

    # appropriately name the model
    modelsDict['logReg{0}'.format(i)] = model

    # add predicted values to the array of predictions
    pred = np.transpose(model.predict(X_test))
    y_pred["Column"+str(i)] = pred

  y_pred.columns = y.columns

  return modelsDict, X, y, X_test, y_test, y_pred
