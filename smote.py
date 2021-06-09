# -*- coding: utf-8 -*-
"""
Synthetic Minority Oversampling Technique (SMOTE)
This function transforms an imbalanced dataset (used for binary classification)
into a balanced dataset by oversampling from the minority class.

Reference: 
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
"""

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE


def smote(df):
  
  if df.columns[[0]] == 'Unnamed: 0':
    df.drop(df.columns[[0]], axis=1, inplace=True) # remove first column

  y = df.Status # labels
  x = df.drop('Status', axis=1) # features

  oversample = SMOTE()
  x_oversampled, y_oversampled = oversample.fit_resample(x, y)
  
  # get sample counts for balanced dataset
  counter = Counter(y_oversampled)
  count = counter[0]

  # regenerate dataframe with new samples
  df = pd.concat([pd.DataFrame(y_oversampled), pd.DataFrame(x_oversampled)], axis=1)
  df.columns = ['Status'] + x.columns.to_list()

  return df, count