<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/49031258/122465135-b89f9c00-cf85-11eb-9e3b-b123e2dcae44.jpg">
</p>

As part of the AI4Good Lab 3-week project creation, we decided to build a tool to diagnose motor speech disorders (dysarthrias) caused by conditions such as Parkinson’s and ALS from user audio input. Refer to `notebook.ipynb` for an in-depth review of the project.

# Command Line Usage
To classify your own audio files from the command line, please follow these instructions.  
1. Record yourself sustaining the /a/ vowel sound for 5 seconds. 
2. Convert the audio file to .wav format.
3. Download this repository and navigate into it from the command line using ```cd```.
4. On the command line, type the command ``` python3 main.py -i <path_to_audio_file> ```
5. For a reminder of the usage, you can type ``` python3 main.py -h ```

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
