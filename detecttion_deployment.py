import json
import base64
import os
from io import BytesIO
import copy
from PIL import Image
import numpy as np
import boto3

import onnx
import onnxruntime

# Resize the image based on the model used for training
# Currently, it's 224
IMG_SIZE = (640, 640)

BUCKET_NAME = os.environ['BUCKET_NAME']
BUCKET_KEY = os.environ['BUCKET_KEY']
MODEL_NAME = "model.onnx"
LABELS_DICT = {0: 'drinking', 1: 'hair and makeup', 2: 'operating the radio', 3: 'reaching behind', 4: 'safe driving', 5: 'talking on the phone', 6: 'talking to passenger', 7: 'texting'}
SAFE_DRIVING_INDEX = 4
boto3_s3 = boto3.client('s3')


def get_image_payload(data):
    image = base64.b64decode(data)
    payload = bytearray(image)
    return payload
    
def preprocess_input(img):
    # Convert bytes to RGB channels
    stream = BytesIO(img)
    img_array = Image.open(stream).convert("RGB")
    img_array = img_array.resize(IMG_SIZE)
    img_transposed = np.transpose(img_array, (2, 0, 1))
    image = np.expand_dims(img_transposed, axis=0)
    image_batch = copy.deepcopy(image)
    image_batch_convert = image_batch.astype(np.float32)
    final_batch = image_batch_convert/255.0
    
    return final_batch


def process_post_request(event):
    # Get the base64 encoded image from body
    params = event['body']
    raw_data = params["image"]
    confidence = params.get("conf", "")
    if not confidence:
        confidence = 0.25
    img_payload = get_image_payload(raw_data)
    processed_img = preprocess_input(img_payload)
    print("success")
   
    # Predict using the onnx model
    #TODO: try to package the model
    temp_location = '/tmp/' + MODEL_NAME
    print("Location", temp_location)
    temp_model_path =  boto3_s3.download_file(BUCKET_NAME, BUCKET_KEY, temp_location )
    session = onnxruntime.InferenceSession(temp_location, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(input_name)
    result = session.run([output_name], {input_name: processed_img})
    output = np.transpose(np.squeeze(result[0]))
    rows = output.shape[0]

    # get the labels except 4
    not_safe_keys = list(LABELS_DICT.keys())
    not_safe_keys.remove(SAFE_DRIVING_INDEX)

    unsafe = False
    for row_index in range(rows):
        classes_scores = output[row_index, 4:]
        max_score = classes_scores.max()
        if max_score > confidence:
            class_index = np.argmax(classes_scores)
            if class_index in not_safe_keys:
                unsafe = True
                break

    if unsafe:
        return {
            "status": "success",
            "data": {"safe": False},
            "error": [None]
        }

    return {
        "status": "success",
        "data": {"safe": True},
        "error": [None]
    }


def lambda_handler(event, context):
        try:
            if event["httpMethod"] == "POST":
                print("In")
                response = process_post_request(event)
                return response
            else:
                print("Out")
                print('Unsupported Method: {}'.format(event['httpMethod']))
                return {
                    "status": "Failure",
                    "data": ""
                }
        except Exception as error:
            message = "unsupported method: error: {}".format(str(error))
            print(message)
            return {
                "status": "Failure",
                "data": ""
            }
