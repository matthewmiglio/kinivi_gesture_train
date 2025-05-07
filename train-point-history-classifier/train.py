import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

"""
===Instructions===
1. Create a CSV file named "point_history.csv" and "point_history_classifier_label.csv" in the same directory as this script.
2. Use add_point_history_data.ipynb to populate this file with custom point-history data
3. Your point_history_classifier_label.csv must contain 1 column, no header, and the labels must be in the same order as the point-history labels in the point_history.csv file.
4. The script will produce a trained model in the "trains" folder.
5. The script will also produce a metadata file in the "metadata" folder, which contains the version number and the label information about this model.
"""


def read_csv(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data


def count_keypoint_classes():
    keypoint_file_path = "train-point-history-classifier/point_history.csv"
    seen_lables = []
    with open(keypoint_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            label = line.split(",")[0].strip()
            if label not in seen_lables:
                seen_lables.append(label)
    return len(seen_lables)


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt="g", square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    # if report:
    # print("Classification Report")
    # print(classification_report(y_test, y_pred))


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


class Metadata:
    def __init__(self, version):
        self.metadata_folder = r"metadata"
        os.makedirs(self.metadata_folder, exist_ok=True)
        self.version = version
        fn = f"metadata_{version}.txt"
        self.metadata_file_path = os.path.join(self.metadata_folder, fn)

    def write_metadata(self):
        labels_count = count_keypoint_classes()
        labels_file_path = (
            "train-point-history-classifier/point_history_classifier_label.csv"
        )
        rows = read_csv(labels_file_path)
        metadata_text = ""
        metadata_text += f"Version: {self.version}\n"
        metadata_text += f"Number of classes: {labels_count}\n"
        metadata_text += f"Labels:\n"
        for row in rows:
            metadata_text += f"\t{row}\n"
        with open(self.metadata_file_path, "w") as f:
            f.write(metadata_text)

        print(f"Wrote metadata for this model to {self.metadata_file_path}")
        print(metadata_text)


train_folder = "train-point-history-classifier/trains"
versioner = Versioning()
VERSION = versioner.get_new_version()
Metadata(VERSION).write_metadata()
dataset = "train-point-history-classifier/point_history.csv"
modeL_save_name = f"point_history_classifier{VERSION}.keras"
model_save_path = os.path.join(train_folder, modeL_save_name)
RANDOM_SEED = 42
PATIENCE = 15
TIME_STEPS = 16
DIMENSION = 2
use_lstm = False
print("=" * 50)
print(f"\n\nTraining version: {VERSION}")
os.makedirs(train_folder, exist_ok=True)
NUM_CLASSES = count_keypoint_classes()
print(f"Number of classes: {NUM_CLASSES}")


def train():
    X_dataset = np.loadtxt(
        dataset,
        delimiter=",",
        dtype="float32",
        usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)),
    )
    y_dataset = np.loadtxt(dataset, delimiter=",", dtype="int32", usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
    )
    model = None

    if use_lstm:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION,)),
                tf.keras.layers.Reshape(
                    (TIME_STEPS, DIMENSION), input_shape=(TIME_STEPS * DIMENSION,)
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(16, input_shape=[TIME_STEPS, DIMENSION]),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=PATIENCE, verbose=1, restore_best_weights=True
    )
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

    model = tf.keras.models.load_model(model_save_path)
    predict_result = model.predict(np.array([X_test[0]]))
    print(np.squeeze(predict_result))
    print(np.argmax(np.squeeze(predict_result)))

    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    print_confusion_matrix(y_test, y_pred)
    model.save(model_save_path, include_optimizer=False)
    model = tf.keras.models.load_model(model_save_path)
    tflite_save_path = model_save_path.replace(".keras", ".tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, "wb").write(tflite_quantized_model)
    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    interpreter.set_tensor(input_details[0]["index"], np.array([X_test[0]]))
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]["index"])
    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))
    print(f"\n\nDone training model. Look for version {VERSION} in the trains folder.")


if __name__ == "__main__":
    train()
