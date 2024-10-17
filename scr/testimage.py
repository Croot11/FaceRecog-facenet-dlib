from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dlib

import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC

# Tắt eager execution
tf.compat.v1.disable_eager_execution()

# Load mô hình và dữ liệu từ file
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

# Tạo session TensorFlow
sess = tf.compat.v1.Session()

# Load mô hình Facenet
facenet.load_model(FACENET_MODEL_PATH)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Đường dẫn đến thư mục chứa ảnh
IMAGE_DIR = r"C:\Users\Acer\OneDrive\Documents\NCKH\datatrain\ngoccc143"
# Lặp qua tất cả các ảnh trong thư mục
total_prob = 0.0
count = 0  # Biến đếm số lượng ảnh
first_best_name = None  # Biến lưu trữ best_name của ảnh đầu tiên
for filename in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is not None:
        # Tiền xử lý ảnh và đưa vào mô hình
        scaled = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        
        scaled_reshape = scaled.reshape(-1, 160, 160, 3)
        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)

        # Dự đoán
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        best_name = class_names[best_class_indices[0]]

        if count == 0:
            first_best_name = best_name
        else:
            # So sánh best_name của ảnh này với first_best_name
            if best_name != first_best_name:
                print("Fail to recognize: Different best_name detected in images.")
                break
        
        # Cộng tổng xác suất cho mỗi tên
        total_prob += best_class_probabilities[0]
        
        count += 1  # Tăng biến đếm số lượng ảnh
    else:
        print("Could not read image:", img_path)

avg_prob = total_prob / count  # Tính toán xác suất trung bình
print("Name: {}, Average Probability: {}".format(best_name, avg_prob))

# Đóng session TensorFlow
sess.close()
