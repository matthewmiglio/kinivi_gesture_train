Kinivi Hand Gesture Recognition Project
Project Description
This project is designed to enhance hand-gesture recognition using models sourced from Kinivi's hand-gesture models. The primary purpose is to improve the accuracy and versatility of hand-gesture recognition by adding new training pipelines, incorporating more data, and defining custom gestures. The project includes both storage-based and live camera feed demonstrations.

Tech Stack
Python Version: 3.11.0

Dependencies:

opencv-python: 4.11.0.86
numpy: 1.26.4
mediapipe: 0.10.21
tensorflow: 2.19.0
scikit-learn: 1.6.1
pandas: 2.2.3
seaborn: 0.13.2
matplotlib: 3.10.1
How to Use
Training Keypoint Classifier
Prepare Keypoint Data:

Use train-keypoint-classifier\add_keypoint_data_to_csv.ipynb to populate keypoint.csv with custom keypoint data.
Ensure keypoint_classifier_label.csv contains the labels in the same order as the keypoints in keypoint.csv.
Train the Keypoint Classifier:

Run train-keypoint-classifier\train.py to train the keypoint detection model.
Example command:
python train-keypoint-classifier\train.py
Copy
Training Point History Classifier
Prepare Point History Data:

Collect videos of the hand gestures.
Use train-point-history-classifier\add_point_history_data.ipynb to convert videos to point history data.
Format the data appropriately for training.
Train the Point History Classifier:

Run train-point-history-classifier\train.py to train the point history model.
Example command:
python train-point-history-classifier\train.py
Copy
Inference Demo Apps
From Storage
Prepare the Demo Data:

Ensure you have images or videos stored locally for inference.
Run the Demo Script:

Use test-kinivi-gesture\storage_prediction.py to perform inference on stored data.
Example command:
python test-kinivi-gesture\storage_prediction.py --input_path /path/to/input/data
Copy
From Camera
Set Up the Camera:

Ensure your camera is properly connected and configured.
Run the Demo Script:

Use test-kinivi-gesture\camera_prediction.py to perform live inference using the camera feed.
Example command:
python test-kinivi-gesture\camera_prediction.py
Copy
Instructions for Setting Up the Project
Ensure Required Files are in Place:

test-kinivi-gesture\model\keypoint_classifier\keypoint_classifier.py and test-kinivi-gesture\model\point_history_classifier\point_history_classifier.py must be populated with valid label files: keypoint_classifier_label.csv and point_history_classifier_label.csv.
Optional Model Training:

You can train your own models using train-point-history-classifier\train.py and train-keypoint-classifier\train.py, and move those files to the model folders.
Recognized Gestures:

The script recognizes POINT_SIGN_ID (3) as a point gesture and will not classify other gestures (e.g., palm, fist, peace sign, OK).
Additional Tools
Label Changer: train-point-history-classifier\isolated_gesture_sets\label_changer.ipynb
Assembler: train-point-history-classifier\isolated_gesture_sets\assembler.ipynb
Analysis: train-point-history-classifier\isolated_gesture_sets\analysis.ipynb
Set Cutter: train-point-history-classifier\isolated_gesture_sets\_set_cutter.ipynb
These tools can be used to create a full point-history dataset.

Example Commands
Training Keypoint Classifier
python train-keypoint-classifier\train.py
Copy
Training Point History Classifier
python train-point-history-classifier\train.py
Copy
Inference from Storage
python test-kinivi-gesture\storage_prediction.py --input_path /path/to/input/data
Copy
Inference from Camera
python test-kinivi-gesture\camera_prediction.py
Copy
Feel free to reach out if you have any questions or need further assistance with the project!
