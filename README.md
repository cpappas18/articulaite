# Goal 
Build a tool to diagnose motor speech disorders caused by conditions such as Parkinson’s and ALS from user audio input.

# How to use

# Code breakdown
`smote_binary(df)` returns an augmented dataframe using the SMOTE technique, as well as its row count.

`smote_multiclass_encoded(df)` returns an augmented dataframe using SMOTE, its row count, and one-hot encodes the classes.

`performance_report(model, X, y, X_test, y_test, y_pred, model_label)` creates a CSV file with various performance metrics and graphs for easy comparison.

`logreg_pred(url)` fits a logistic regression model and returns the necessary parameters for `performance_report`.

`feature_extraction.ipynb` contains the code for extracting sound features relevant to dysarthria diagnosis from audio files.

# Team
Chloe Pappas, <chloeoliviapappas@gmail.com>  
Hala Hassan, <halahassan13@gmail.com>  
Nadia Enhaili, <nadia.enhaili@gmail.com>  
Ritu Ataliya, <atal2950@mylaurier.ca>  
Jiayue Yang, <jiayue.yang@mail.mcgill.ca>  
Kamun Karl Itaj, <kamun.karl-itaj@hotmail.ca>  

# Acknowledgements


 
# References
[SMOTE technique](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
