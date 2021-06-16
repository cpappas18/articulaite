# Goal 
As part of the AI4Good Lab 3-week project creation, we decided to build a tool to diagnose motor speech disorders (dysarthrias) caused by conditions such as Parkinson’s and ALS from user audio input. Refer to `notebook.ipynb` for an in-depth review of the project.

# File explorer
` ` ` feature_extraction.ipynb ` ` ` 

Contains the code for extracting sound features relevant to dysarthria diagnosis from audio files.

 ` ` ` smote.py ` ` `

Contains two functions, `smote_binary` and `smote_multiclass`, that oversample or undersample a dataframe using the SMOTE technique and one-hot encode it.

` ` ` performance_report.py ` ` `

Contains an eponymous function that generates a CSV file with various performance metrics and graphs for easy comparison of models.

` ` ` logistic_regression.py ` ` `

Contains a function that fits a logistic regression model for each of the specified n classes (default is 1) and returns the necessary parameters for `performance_report`.


# Team
Chloe Pappas, <chloeoliviapappas@gmail.com>  
Hala Hassan, <halahassan13@gmail.com>  
Nadia Enhaili, <nadia.enhaili@gmail.com>  
Ritu Ataliya, <atal2950@mylaurier.ca>  
Jiayue Yang, <jiayue.yang@mail.mcgill.ca>  
Kamun Karl Itaj, <kamun.karl-itaj@hotmail.ca>  

# Acknowledgements
Special thanks to our TA, Nadia Blostein, for her invaluable guidance and clever insights throughout the program !  
Thank you to our team mentors Ainaz, Disha and Isabelle for sharing their expertise and weekly consulting meetings !  
Thank you to the AI4Good Lab team for creating this opportunity in the first place !

# References
[SMOTE technique](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
