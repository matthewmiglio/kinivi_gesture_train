import numpy as np
import tensorflow as tf

def keypoint_infer(landmark_list):
    """
    Runs inference on a list of 21 hand landmarks using a TFLite keypoint classifier,
    and returns the corresponding gesture label.

    Args:
        landmark_list (list of [x, y]): A list of 21 (x, y) hand landmark coordinates.

    Returns:
        str: The predicted gesture label.
    """
    # Output label mapping
    labels = [
        "open",
        "close",
        "camera_point",
        "vert_point"
    ]

    def pre_process_landmark(landmark_list):
        """Normalize landmarks relative to the wrist and scale to unit range."""
        temp_landmark_list = landmark_list.copy()
        base_x, base_y = temp_landmark_list[0]

        # Translate relative to wrist
        for i in range(len(temp_landmark_list)):
            temp_landmark_list[i][0] -= base_x
            temp_landmark_list[i][1] -= base_y

        # Flatten
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalize
        max_val = max(map(abs, temp_landmark_list)) or 1.0
        temp_landmark_list = [coord / max_val for coord in temp_landmark_list]
        return temp_landmark_list

    import itertools
    input_data = pre_process_landmark(landmark_list)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="your_keypoint_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input
    input_array = np.array([input_data], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_array)

    # Run inference
    interpreter.invoke()

    # Get result
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output))

    return labels[predicted_index]
