Project Description
This project is designed to enhance hand-gesture recognition using models sourced from Kinivi's hand-gesture models. The primary purpose is to improve the accuracy and versatility of hand-gesture recognition by adding new training pipelines, incorporating more data, and defining custom gestures. Additionally, the demo file has been modified to support both storage-based and live camera feed demonstrations.

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
a) Training Keypoint
Prepare the Dataset:

Ensure you have a dataset of images with hand gestures.
Annotate the keypoints for each gesture using a tool like LabelImg.
Run the Training Script:

Use the provided training script to train the keypoint detection model.
Example command:
python train_keypoint.py --dataset_path /path/to/dataset --output_model /path/to/save/model
Copy
b) Training Point History
Prepare the Point History Data:

Collect the point history data for various gestures.
Format the data appropriately for training.
Run the Training Script:

Use the provided training script to train the point history model.
Example command:
python train_point_history.py --data_path /path/to/point/history/data --output_model /path/to/save/model
Copy
c) Inference Demo Apps
From Storage
Prepare the Demo Data:

Ensure you have images or videos stored locally for inference.
Run the Demo Script:

Use the provided demo script to perform inference on stored data.
Example command:
python demo_from_storage.py --input_path /path/to/input/data --model_path /path/to/trained/model
Copy
From Camera
Set Up the Camera:

Ensure your camera is properly connected and configured.
Run the Demo Script:

Use the provided demo script to perform live inference using the camera feed.
Example command:
python demo_from_camera.py --model_path /path/to/trained/model
Copy
Feel free to reach out if you have any questions or need further assistance with the project!