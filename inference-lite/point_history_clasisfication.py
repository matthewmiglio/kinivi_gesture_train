import numpy as np
import tensorflow as tf

def point_history_infer(input_data):
    """
    Runs inference on a list of 32 float values using a TFLite gesture classification model,
    and returns the corresponding gesture label.

    Args:
        input_data (list of float): A list of 32 float values representing point history.

    Returns:
        str: The predicted gesture label from the predefined set.
    """
    # Define output label mapping
    labels = [
        "stop",
        "cw_1",
        "ccw_2",
        "erratic",
        "downswipe",
        "upswipe",
        "leftswipe",
        "rightswipe"
    ]

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input as float32 array
    input_array = np.array([input_data], dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor and determine predicted index
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output))

    # Return corresponding label
    return labels[predicted_index]
