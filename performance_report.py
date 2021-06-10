import csv
import matplotlib.pyplot as plt
import os
import secrets  # to generate jpg name

def performance_report(model, X, y, X_test, y_test, y_pred, model_label):

  # classification accuracy
  try:
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
  except:
    accuracy = "NA"

  # confusion matrix
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import plot_confusion_matrix

  try:
    conf = confusion_matrix(y_test, y_pred)
    TP = conf[1][1]
    TN = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]

    disp = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, values_format='.0f')
    title = secrets.token_hex(4)
    plt.savefig(title, format="jpg")

    path1 = os.path.abspath(title+".jpg")
  except:
    TP = TN = FP = FN = path1 = ""

  # k-fold cross validation
  from sklearn import model_selection
  try:
    kfold = model_selection.KFold(n_splits=6, shuffle=True)
    KCV = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
  except:
    KCV = ""

  # log loss (closer to 0 is better)
  try:
    logloss = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='neg_log_loss').mean()
  except:
    logloss = ""

  # AUC (1 is perfect predictions, 0.5 is as good as random)
  try:
    AUC = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='roc_auc').mean()
  except:
    AUC = ""

  # precision recall curve
  from sklearn.metrics import plot_precision_recall_curve
  from sklearn.metrics import average_precision_score

  try:
    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)

    disp = plot_precision_recall_curve(model, X_test, y_test)
    title = secrets.token_hex(4)  
    plt.savefig(title, format="jpg")

    path2 = os.path.abspath(title+".jpg")
  except:
    average_precision = path2 = ""

  # appending metrics to csv file
  with open('metrics.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    if os.stat("metrics.csv").st_size == 0:
      writer.writerow(["Model label", "Accuracy score", "TP", "TN", "FP", "FN", "Confusion matrix file","KCV", "Log-loss", "AUC", "Average precision", "Precision-recall file"])
    writer.writerow([model_label, accuracy, TP, TN, FP, FN, path1, KCV, logloss, AUC, average_precision, path2])
