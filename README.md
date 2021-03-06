<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/49031258/122465135-b89f9c00-cf85-11eb-9e3b-b123e2dcae44.jpg">
</p>

Dysarthria is a motor speech disorder that arises from weakness or paralysis of muscles in the face, lips, tongue, and throat. It is caused by neurological damage and is often one of the first symptoms of numerous common neurological disorders. Dysarthria affects 70 - 100% of people with Parkinson’s disease, 30% of people with ALS (Lou Gehrig's disease), and 20% of people with cerebral palsy. 

Diagnosis of these disorders require MRI and CT scans, blood and urine tests, and EEG or electromyography tests. These tests are very expensive and may be inaccessible for some people. For our final project at the AI4Good Lab, we decided to build a machine learning tool that is capable of detecting a patient's underlying cause of dysarthria by classifying audio input as being indicative of Parkinson's disease, ALS, or cerebral palsy. The best part is that this tool is free and accessible to all!

Refer to [`project_journal.ipynb`](https://github.com/cpappas18/articulaite/blob/main/project_journal.ipynb) for an in-depth review of the data collection, data visualization, and machine learning model.

# Table of contents
1. [How to use](https://github.com/cpappas18/articulaite#how-to-use)
2. [File description](https://github.com/cpappas18/articulaite#file-description)
3. [The team](https://github.com/cpappas18/articulaite#the-team)
4. [Acknowledgements](https://github.com/cpappas18/articulaite#acknowledgements)
5. [References](https://github.com/cpappas18/articulaite#references)

# How to use
To classify your own audio files from the command line, please follow these instructions:  
1. Record yourself sustaining the /a/ vowel sound for 5 seconds. 
2. Convert the audio file to .wav format.
3. Download this repository and navigate into it from the command line using ```cd```
4. To install the necessary packages, type ```pip install -r requirements.txt```
5. To classify your audio file, type the command ``` python3 main.py -i <path_to_audio_file> ```
6. For a reminder of the usage, you can type ``` python3 main.py -h ```

# File description
``` classifiers/ ```

A directory containing all of the machine learning models that we tried for this task. This includes decision tree, multi-layer perceptron, random forest, SVM, and logistic regression models.

``` audio_features_visualization.ipynb ```

Contains code that retrieves audio input from the user and performs visualization of this input, such as plotting spectrograms and intensity graphs. 

``` feature_extraction.py ``` 

Contains code for extracting sound features relevant to Dysarthria diagnosis from a .wav audio file.

``` main.py ```

Main function for taking audio input from the command-line and predicting a class of Parkinson's, ALS, Cerebral Palsy, or healthy.

``` project_journal.ipynb ```

Contains a summary of our data collection, preprocessing, augmentation, model development, and model selection process.

``` performance_report.py ```

Contains a function that generates a CSV file with various performance metrics and graphs for easy comparison of machine learning models.

``` random_forest_final_model.sav ```

The trained model we ended up using for our classification.

``` smote.py ```

Contains two functions, `smote_binary` and `smote_multiclass`, that oversample and/or undersample binary class and multiclass dataframes using the SMOTE technique.

# The team
Chloe Pappas, <chloeoliviapappas@gmail.com>  
Hala Hassan, <halahassan13@gmail.com>  
Nadia Enhaili, <nadia.enhaili@gmail.com>  
Ritu Ataliya, <atal2950@mylaurier.ca>  
Jiayue Yang, <jiayue.yang@mail.mcgill.ca>  
Kamun Karl Itaj, <kamun.karl-itaj@hotmail.ca>  

# Acknowledgements
We are very thankful for our TA, Nadia Blostein, for her invaluable guidance and clever insights throughout the program. Thank you to our team mentors Ainaz, Disha, and Isabelle for sharing their expertise and meeting with us for weekly consultations. Finally, thank you to the AI4Good Lab for creating this opportunity and providing us with the education and materials necessary to develop this project!

# References
[SMOTE technique](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
