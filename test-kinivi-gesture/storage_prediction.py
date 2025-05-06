import random
import csv
import copy
import argparse
import itertools
import os
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier
from model import PointHistoryClassifier

"""
===Instructions===
1. Make sure test-kinivi-gesture\test-kinivi-gesture\model\keypoint_classifier\keypoint_classifier.py and test-kinivi-gesture\test-kinivi-gesture\model\point_history_classifier\point_history_classifier.py 
are populated with valid label files: keypoint_classifier_label.csv & point_history_classifier_label.csv
2. Optionally, you can train your own models using train-point-history-classifier\train.py & train-keypoint-classifier\train.py, and move those files to the model folders.
3. The script only recognizes POINT_SIGN_ID (3) as a point gesture, and will not point-classify other gestures (e.g. palm, fist, peace sign, OK).
"""

USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
LOOP = True
POINT_SIGN_ID = 3  # vertical point


def read_keypoint_classifier_labels():
    with open(
        "test-kinivi-gesture\model\keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        return keypoint_classifier_labels


def read_point_history_classifier_labels():
    with open(
        "test-kinivi-gesture\model\point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
        return point_history_classifier_labels


def test_gesture_from_video_or_folder(
    path,
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    keypoint_classifier = KeyPointClassifier()
    keypoint_classifier_labels = read_keypoint_classifier_labels()
    point_history_classifier = PointHistoryClassifier()
    point_history_classifier_labels = read_point_history_classifier_labels()
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # load the images depending if folder or video file
    if os.path.isdir(path):
        print(f"Processing this path as a folder: {path}")
        image_files = sorted([f for f in os.listdir(path)])
        image_paths = [os.path.join(path, image_file) for image_file in image_files]
        images = [cv.imread(image_path) for image_path in image_paths]
    elif os.path.isfile(path):
        print(f"Processing this path as a video file: {path}")
        images = avi_to_images(path)
    else:
        print(f"WARNING! This path is neither a folder nor a file: {path}")
        return

    while True:
        # play video once
        for image in images:
            key = cv.waitKey(10)
            if key == 27:  # ESC
                return

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history, desired_length=16
                    )

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_sign_string = hand_sign_id_to_string(hand_sign_id)
                    print(f"Current hand sign string")

                    if hand_sign_id == POINT_SIGN_ID:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list
                        )

                    current_finger_gesture_string = finger_gesture_id_to_string(
                        finger_gesture_id
                    )
                    print(f"Current finger gesture: {current_finger_gesture_string}")

                    finger_gesture_history.append(finger_gesture_id)
                    finger_gesture_history_count_info = Counter(
                        finger_gesture_history
                    ).most_common()
                    total_history_count = len(finger_gesture_history)
                    most_common_count = finger_gesture_history_count_info[0][1]
                    most_common_gesture_occurence_ratio = round(
                        (100 * (most_common_count / total_history_count)), 2
                    )

                    dominant_finger_gesture_id = finger_gesture_history_count_info[0][0]
                    domaint_finger_gesture_string = finger_gesture_id_to_string(
                        dominant_finger_gesture_id
                    )

                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[dominant_finger_gesture_id],
                    )
                    print(f"Dominant finger gesture: {domaint_finger_gesture_string}")

            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, 999, 0, -999)

            cv.imshow("Hand Gesture Recognition", debug_image)

        if not LOOP:
            break

    cv.destroyAllWindows()


def hand_sign_id_to_string(id):
    index2label = {}
    csv_path = r"test-kinivi-gesture\model\keypoint_classifier\keypoint_classifier_label.csv"
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        for i, row in enumerate(rows):
            index2label[i] = str(row)

    return index2label[id]


def finger_gesture_id_to_string(id):
    index2label = {}
    csv_path = r"test-kinivi-gesture\model\point_history_classifier\point_history_classifier_label.csv"
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        for i, row in enumerate(rows):
            index2label[i] = str(row)

    return index2label[id]


def avi_to_images(video_file):
    images = []
    cap = cv.VideoCapture(video_file)
    frame_count = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        images.append(image)

        frame_count += 1
    cap.release()
    return images


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history, desired_length):
    if len(point_history) < desired_length:
        # print(f'WARNING! Point history is too small t preprocess! Returning empty list')
        return []

    elif len(point_history) > desired_length:
        possible_indices = list(range(len(point_history)))
        random_selected_indicies = random.sample(possible_indices, desired_length)
        sorted_random_indicies = sorted(random_selected_indicies)
        point_history = [point_history[i] for i in sorted_random_indicies]

    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, logging_mode, landmark_list, point_history_list):
    writing_mode = False
    print_mode = False
    skip_mostly_zeros = True
    max_zero_count = 10

    if skip_mostly_zeros:
        zero_count = 0
        for item in point_history_list:
            if item in [0, 0.0, "0"]:
                zero_count += 1
        if zero_count > max_zero_count:
            return

    if logging_mode == 0:
        pass
    if logging_mode == 1:
        csv_path = "test-kinivi-gesture\model\keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if writing_mode is True:
                writer.writerow([number, *landmark_list])

            row_content = [number, *landmark_list]
            if print_mode:
                print("Writing this content to keypoint.csv: ", row_content)
    if logging_mode == 2:
        if len(point_history_list) != 32:
            if print_mode:
                print(f"Point history is incomplete... not logging this line!")
            return
        csv_path = "test-kinivi-gesture\model\point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if writing_mode is True:
                writer.writerow([number, *point_history_list])

            row_content = [number, *point_history_list]
            if print_mode:
                print("Writing this content to point_history.csv: ", row_content)
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
            )

    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = ["Logging Key Point", "Logging Point History"]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image


if __name__ == "__main__":
    gesture_video = r"C:\my_files\data\matt_gesture\labeled_videos\erratic\WIN_20250502_20_17_38_Pro.mp4"
    test_gesture_from_video_or_folder(
        gesture_video,
    )
