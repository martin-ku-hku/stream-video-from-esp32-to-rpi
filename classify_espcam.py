# python3

"""Example using TF Lite to classify objects with the ESP32-cam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np

import requests
from io import BytesIO
import cv2

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate



def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)


  interpreter = Interpreter(args.model) # Use this line if you don't use an EdgeTPU
  # interpreter = Interpreter(args.model, experimental_delegates=[load_delegate('libedgetpu.so.1')]) # Use this line if you have an EdgeTPU
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  ip_addr = '192.168.2.29'
  stream_url = 'http://' + ip_addr + ':81/stream'  
  res = requests.get(stream_url, stream=True)
  for chunk in res.iter_content(chunk_size=100000):
    if len(chunk) > 100:
      try:
        start_time = time.time()
        img_data = BytesIO(chunk)
        cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
        cv_resized_img = cv2.resize(cv_img, (width, height), interpolation = cv2.INTER_AREA)
        results = classify_image(interpreter, cv_resized_img)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        cv2.putText(cv_img, f'{labels[label_id]}', (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(cv_img, f'{prob}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imshow("OpenCV", cv_img)
        cv2.waitKey(1)
        print(f'elapsed_ms: {elapsed_ms}')
      except Exception as e:
        print(e)
        continue

if __name__ == '__main__':
  main()
