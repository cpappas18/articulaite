#!/usr/bin/python

import sys
import getopt
import os
import pickle
import numpy as np
from feature_extraction import extract_features


def main(argv):
    audio_file = ''

    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="]) # parsing inputs
    except getopt.GetoptError: # error with command-line inputs
        print("Usage: main.py -i <audio_file>")
        sys.exit(2)

    if len(opts) == 0:
        print("Usage: main.py -i <audio_file>")
        sys.exit(2)
    else:
        for opt, arg in opts:
            if opt == "-h": # help option
                print("Usage: main.py -i <audio_file>")
                sys.exit()
            elif opt in ["-i", "--ifile"]:
                path, extension = os.path.splitext(arg) # setting audio file based on input
                audio_file = arg

    if os.path.exists(audio_file) and extension == '.wav':
        print(f"Audio file is {audio_file}")
    elif not os.path.exists(audio_file):
        print("Error: Audio file does not exist. Please give the complete path.")
        print("Usage: main.py -i <audio_file>")
        sys.exit(1)
    elif extension != '.wav':
        print("Error: Audio file must have format .wav.")
        sys.exit(1)
    else:
        print("Error.")
        print("Usage: main.py -i <audio_file>")
        sys.exit(1)

    while True:
        gender = input("Please input your gender (M/F): ")
        if gender in ["F", "f", "female"]:
            gender = 0
            break
        elif gender in ["M", "m", "male"]:
            gender = 1
            break
        else:
            print("Error with gender input. Please try again.")

    # get features from audio file
    features = extract_features(audio_file)
    input_features = [gender, features['localJitter'], features['localAbsJitter'], features['ppq5'], features['ddp'],
                   features['localShimmer'], features['localdbShimmer'], features['apq3Shimmer'], features['apq5Shimmer'],
                   features['apq11Shimmer'], features['DFA']]

    # remove NaNs
    for i in range(len(input_features)):
        if input_features[i] is None:
            input_features[i] = 0

    feature_means = [0.335300, 0.005682, 0.000041, 0.003049, 0.008125, 0.037442, 0.340817, 0.019002, 0.022342, 0.030332, 0.655004]
    feature_std_dev = [0.472128, 0.057122, 0.056844, 0.056964, 0.057615, 0.064179, 0.271675, 0.058891, 0.059949, 0.061402, 0.102698]

    # standardize features
    for i in range(len(input_features)):
        if i>0: # do not standardize gender
            feature = input_features[i]
            std_feature = (feature-feature_means[i])/feature_std_dev[i]
            input_features[i] = std_feature

    # predict class
    with open('random_forest_final_model.sav', 'rb') as f:
        model = pickle.load(f)

    input_features = np.array(input_features)
    prediction = model.predict(input_features.reshape(1, -1)).item()

    class_labels = {0: "healthy", 1: "Parkinson's disease", 2: "ALS", 3: "Cerebral Palsy"}

    if prediction == 0:
        print("The model predicts that you do not have Dysarthria - you are healthy!")
    else:
        print(f"The model predicts that you have {class_labels[prediction]}. "
              f"This is not a formal diagnosis. Please visit a doctor immediately for further testing.")


if __name__ == "__main__":
    main(sys.argv[1:])
