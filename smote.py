# -*- coding: utf-8 -*-
"""
## Synthetic Minority Oversampling Technique (SMOTE)
These functions transform an imbalanced dataset into a balanced dataset by using the SMOTE technique to
oversample from the minority class(es).

Reference: 
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
"""

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where, random


"""
Performs SMOTE on a dataframe with binary labels.
"""
def smote_binary(df):
  
  if df.columns[[0]] == 'Unnamed: 0':
    df.drop(df.columns[[0]], axis=1, inplace=True) # remove first column

  y = df.Status # labels
  x = df.drop(['Status', 'Gender'], axis=1) # features only, Gender also removed because it must be binary
  gender = df.Gender

  oversample = SMOTE()
  x_oversampled, y_oversampled = oversample.fit_resample(x, y)

  # get sample counts for balanced dataset
  counter = Counter(y_oversampled) # {label_0: num_0, label_1: num_1}
  if counter[0] == counter[1]: 
    count = counter[0]
  else:
    print("Oversampling failed")

  # regenerate dataframe with new samples
  df_concat = pd.concat([pd.DataFrame(y_oversampled), pd.DataFrame(gender), pd.DataFrame(x_oversampled)], axis=1)
  df_concat.columns = df.columns

  # fill missing gender values with a random choice of 0 or 1
  for row in df_concat.loc[df_concat.Gender.isnull(), 'Gender'].index:
    df_concat.at[row, 'Gender'] = random.randint(0, 2)

  return df_concat, count


"""
Performs SMOTE on a dataframe with multi-class one-hot encoded labels. 
"""
def smote_multiclass_encoded(df):

  if df.columns[[0]] == 'Unnamed: 0':
    df.drop(df.columns[[0]], axis=1, inplace=True)  # remove first column

  y = df.values[:, 0:4]  # one-hot encoded labels
  x = df.values[:, 5:]  # features only, Gender also removed because it must be binary
  gender = df.Gender

  oversample = SMOTE()
  x_oversampled, y_oversampled = oversample.fit_resample(x, y)

  numeric_y_os = [np.where(r == 1)[0][0] for r in y_oversampled]

  # get sample counts for balanced dataset
  counter = Counter(numeric_y_os)
  if counter[0] == counter[1] and counter[1] == counter[2] and counter[2] == counter[3]:
    count = counter[0]
  else:
    print("Oversampling failed")

  # regenerate dataframe with new samples
  df_concat = pd.concat([pd.DataFrame(y_oversampled), pd.DataFrame(gender), pd.DataFrame(x_oversampled)], axis=1)
  df_concat.columns = df.columns

  # fill missing gender values with a random choice of 0 or 1
  for row in df_concat.loc[df_concat.Gender.isnull(), 'Gender'].index:
    df_concat.at[row, 'Gender'] = random.randint(0, 2)

  return df_concat, count
