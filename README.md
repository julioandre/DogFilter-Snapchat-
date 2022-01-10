# DogsFilter-Snapchat-
Using Computer Vision to Build an Emotion-Based Dog Filter in Python 3
This repo contains the code for a Sign Language Classifier. It uses computer vision and python to apply a snapchat filter to a face based on emotion
and using the webcam to record live video as input

# Dependencies and Libraries
* OpenCV
* NumPy
* PyTorch

# Folder Structure/Code Explanation 
* Data Folder: This contains all the data sets we used in training and testing our models
* venv: This folder contains the python environment files to help us create a virtual environment to run the code
* step_2_face_detect.py: This file contains code for detecting faces from an image
* step_3_camera_face_detect.py: This file contains the code for connecting the camera to detect a face
* step_5_ls_simple.py, step_7_fer.py: This file contains code to build a model to dtect face emotions
* step_8_dog_emotion_mask.py: This file contains code to apply the two different filters based on if one is smiling or not 

# Running the application

To run this application we first need to activate the virtual python environment by running the command $source signlanguage/bin/activate
Finally to run the application type this command into your terminal $python step_8_dog_emotion_mask.py   




