import csv
import os.path
from os import path
from sklearn.utils.testing import ignore_warnings # remove annoying warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def performance_report(model, X_train, X_test, y_train, y_test, y_pred):

  # classification accuracy
  try:
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
  except:
    print('Couldnt compute classification accuracy.')

  # confusion matrix
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import plot_confusion_matrix

  try:
    conf = confusion_matrix(y_test, y_pred)
    TP = conf[1][1]
    TN = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]

    disp = plot_confusion_matrix(logReg, X_test, y_test, cmap=plt.cm.Blues, values_format='.0f')
    disp.ax_.set_title('Confusion matrix')
    plt.show()
  except:
    print('Couldnt compute confusion matrix.')

  # k-fold cross validation
  from sklearn import model_selection
  try:
    kfold = model_selection.KFold(n_splits=6, shuffle=True)
    KCV = model_selection.cross_val_score(logReg, X_train, 
                                          y_train, cv=kfold, scoring='accuracy').mean()
  except:
    print('Couldnt compute KCV.')

  # log loss (closer to 0 is better)
  try:
    logloss = model_selection.cross_val_score(logReg, X, y, 
                                              cv=kfold, scoring='neg_log_loss').mean()
  except:
    print('Couldnt compute log-loss.')

  # AUC (1 is perfect predictions, 0.5 is as good as random)
  try:
    AUC = model_selection.cross_val_score(logReg, X, y, cv=kfold, scoring='roc_auc').mean()
  except:
    print('Couldnt compute AUC.')

  # precision recall curve
  from sklearn.metrics import plot_precision_recall_curve
  from sklearn.metrics import average_precision_score

  try:
    y_score = logReg.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)

    disp = plot_precision_recall_curve(logReg, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                      'AP={0:0.2f}'.format(average_precision))
  except:
    print('Couldnt compute precision and recall curve.')

  # appending metrics to csv file
  with open('metrics.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer = csv.writer(file)
    if os.stat("metrics.csv").st_size == 0:
      print('Creating metrics.csv')
      writer.writerow(["Accuracy score", "TP", "TN", "FP", "FN", "KCV", "Log-loss", "AUC"])
    writer.writerow([accuracy, TP, TN, FP, FN, KCV, logloss, AUC])
