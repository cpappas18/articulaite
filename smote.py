# -*- coding: utf-8 -*-

## Synthetic Minority Oversampling Technique (SMOTE)
This script demonstrates how to transform an imbalanced dataset (used for binary classification) into a balanced dataset by oversampling from the minority class.

Reference: 
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
"""

import pandas as pd
import numpy as np
import imblearn
from random import random
from collections import Counter
from matplotlib import pyplot
from numpy import where
from imblearn.over_sampling import SMOTE

def smote(df):
  
  if df.columns[[0]] == 'Unnamed: 0':
    df.drop(df.columns[[0]], axis=1, inplace=True) # remove first column

  y = df.Status # labels
  x = df.drop(['Status', 'Gender'], axis=1) # features only, Gender also removed because it must be binary
  gender = df.Gender

  counter = Counter(y) # {label_0: num_0, label_1: num_1}
  if counter[0] < counter[1]:
    minority = counter[0]
    majority = counter[1]
  else:
    minority = counter[1]
    majority = counter[0]

  oversample = SMOTE()
  x_oversampled, y_oversampled = oversample.fit_resample(x, y)

  # fill gender column with random choice of 0 or 1
  gender_oversampled = gender.to_list()
  for _ in range(majority-minority):
    rand = random()
    if rand < 0.5:
      gender_oversampled.append(0)
    else:
      gender_oversampled.append(1)

  # get sample counts for balanced dataset
  counter = Counter(y_oversampled) # {label_0: num_0, label_1: num_1}
  if counter[0] == counter[1]: 
    count = counter[0]
  else:
    print("Oversampling failed")

  # regenerate dataframe with new samples
  df = pd.concat([pd.DataFrame(y_oversampled), pd.DataFrame(gender_oversampled), pd.DataFrame(x_oversampled)], axis=1)
  df.columns = ['Status', 'Gender'] + x.columns.to_list()

  return df, count
