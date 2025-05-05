import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

"""
===Instructions===
1. Create a CSV file named "keypoint.csv" and "keypoint_classifier_label.csv" in the same directory as this script.
2. Use add_keypoint_data_to_csv.ipynb to populate this file with custom keypoint data
3. Your keypoint_classifier_label.csv must contain 1 column, no header, and the labels must be in the same order as the keypoints in the keypoint.csv file.
4. The script will produce a trained model in the "trains" folder.
"""


def count_keypoint_classes():
    keypoint_file_path = "keypoint.csv"
    seen_lables = []
    with open(keypoint_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            label = line.split(",")[0].strip()
            if label not in seen_lables:
                seen_lables.append(label)
    return len(seen_lables)


class Versioning:
    def __init__(self):
        self.version_file_path = r"version.txt"
        if not os.path.exists(self.version_file_path):
            with open(self.version_file_path, "w") as f:
                f.write("0")

    def read_version(self):
        with open(self.version_file_path, "r") as f:
            version = int(f.read())
        return version

    def increment_version(self):
        current_version = self.read_version()
        new_version = current_version + 1
        with open(self.version_file_path, "w") as f:
            f.write(str(new_version))

    def get_new_version(self):
        current_version = self.read_version()
        new_version = current_version + 1
        self.increment_version()
        return new_version


os.makedirs("trains", exist_ok=True)
versioning = Versioning()
version = versioning.get_new_version()
print(f"\nTraining keypoint model: Version #{version}\n")
RANDOM_SEED = 42
dataset = "keypoint.csv"
model_save_path = f"trains/keypoint_classifier{version}.keras"
tflite_save_path = f"trains/keypoint_classifier{version}.tflite"
NUM_CLASSES = count_keypoint_classes()
print(f"\nNumber of classes: {NUM_CLASSES}\n")
X_dataset = np.loadtxt(
    dataset, delimiter=",", dtype="float32", usecols=list(range(1, (21 * 2) + 1))
)
y_dataset = np.loadtxt(dataset, delimiter=",", dtype="int32", usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False
)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback],
)
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
model = tf.keras.models.load_model(model_save_path)
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))



def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt="g", square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print("Classification Report")
        print(classification_report(y_test, y_pred))


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)
model.save(model_save_path, include_optimizer=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, "wb").write(tflite_quantized_model)
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], np.array([X_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]["index"])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
print(f"Look for model version {version} in trains folder")
