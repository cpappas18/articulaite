<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/49031258/122465135-b89f9c00-cf85-11eb-9e3b-b123e2dcae44.jpg">
</p>

Dysarthria is a motor speech disorder that arises from weakness or paralysis of muscles in the face, lips, tongue, and throat. It is caused by neurological damage and is often one of the first symptoms of numerous common neurological disorders. For instance, Dysarthria affects 70-100% of people with Parkinson’s disease, 30% of people with ALS (Lou Gehrig's disease), and 20% of people with Cerebral Palsy. 

Diagnosis of these disorders require MRI and CT scans, blood and urine tests, and EEG or electromyography tests. These tests are very expensive and may be inaccessible for some patients. For our final project at the AI4Good Lab, we decided to build a machine learning tool that is capable of detecting a patient's underlying cause of Dysarthria by classifying audio input as being indicative of Parkinson's disease, ALS, or Cerebral Palsy. The best part is that this tool is free and accessible for all!

Refer to `notebook.ipynb` for an in-depth review of the data collection, data visualization, and machine learning model.

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
