# 🤖 Kinivi Hand Gesture Recognition Project

Enhance hand gesture recognition using Kinivi’s models. This project improves accuracy and flexibility by supporting custom gestures, new training pipelines, and both live and storage-based demos.

---

## 📚 Project Overview

The Kinivi Hand Gesture Recognition Project is built to:

- Improve gesture detection accuracy
- Allow training on **custom gestures**
- Support **live camera feed** and **offline video/image** inference
- Utilize a full suite of tools for gesture dataset management and analysis

---

## 🧰 Tech Stack

- **Python**: `3.11.0`
- **Dependencies**:
  - `opencv-python`: `4.11.0.86`
  - `numpy`: `1.26.4`
  - `mediapipe`: `0.10.21`
  - `tensorflow`: `2.19.0`
  - `scikit-learn`: `1.6.1`
  - `pandas`: `2.2.3`
  - `seaborn`: `0.13.2`
  - `matplotlib`: `3.10.1`

---

## 🧠 How to Use

### ✍️ Training Keypoint Classifier

#### 1. Prepare Keypoint Data
- Use: `train-keypoint-classifier/add_keypoint_data_to_csv.ipynb`
- Populate: `keypoint.csv`
- Label file: `keypoint_classifier_label.csv` (ensure label order matches keypoints)

#### 2. Train Model
```bash
python train-keypoint-classifier/train.py
```

---

### 📈 Training Point History Classifier

#### 1. Prepare Point History Data
- Collect gesture videos
- Use: `train-point-history-classifier/add_point_history_data.ipynb`
- Format data correctly for training

#### 2. Train Model
```bash
python train-point-history-classifier/train.py
```

---

### 🎥 Inference Demo

#### From Storage
- Script: `test-kinivi-gesture/storage_prediction.py`
```bash
python test-kinivi-gesture/storage_prediction.py --input_path /path/to/input/data
```

#### From Camera
- Script: `test-kinivi-gesture/camera_prediction.py`
```bash
python test-kinivi-gesture/camera_prediction.py
```

---

## 🛠️ Project Setup Instructions

- Ensure the following files exist:
  - `test-kinivi-gesture/model/keypoint_classifier/keypoint_classifier.py`
  - `test-kinivi-gesture/model/point_history_classifier/point_history_classifier.py`
  - Label files: 
    - `keypoint_classifier_label.csv`
    - `point_history_classifier_label.csv`

- Optional: Retrain models using the provided scripts and move them into the model folders.

---

## ✋ Recognized Gestures

- The system recognizes `POINT_SIGN_ID = 3` as a valid gesture.
- Other gestures (e.g. ✋ Palm, ✊ Fist, ✌️ Peace, 👌 OK) are currently not classified unless customized.

---

## 🧰 Additional Tools

Located in `train-point-history-classifier/isolated_gesture_sets/`:

| Tool                 | Description                                       |
|----------------------|---------------------------------------------------|
| `label_changer.ipynb` | Change or relabel gestures in the dataset         |
| `assembler.ipynb`     | Combine multiple gesture datasets                 |
| `analysis.ipynb`      | Analyze dataset distribution                      |
| `_set_cutter.ipynb`   | Slice gesture data into training sets             |

---

## 💡 Example Commands

```bash
# Train keypoint classifier
python train-keypoint-classifier/train.py

# Train point history classifier
python train-point-history-classifier/train.py

# Run inference on stored media
python test-kinivi-gesture/storage_prediction.py --input_path /path/to/input/data

# Run live camera gesture recognition
python test-kinivi-gesture/camera_prediction.py
```

---

## ❓ Need Help?

Feel free to [open an issue](https://github.com/) or reach out with questions! Happy coding! 👋
