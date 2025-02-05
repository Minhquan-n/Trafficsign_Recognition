import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

# Function balance() create new image from old image by make blur.
def balance(path, threshold):
    images = os.listdir(path)
    count = len(images)
    if count >= threshold:
        return 'Done'
    image = 0
    while count < threshold:
        img = cv2.imread(path+'/'+images[image])
        k_size = 9
        for _ in range(7):
            gaussian = cv2.GaussianBlur(img, (k_size, k_size), 0)
            cv2.imwrite(path+f'/img_gb{count}.jpg', gaussian)
            count += 1
            k_size += 4
        image += 1

    return 'Done'

# Function saveData() write an array of image to data file
def saveData(target_path, images, label):
    pixels = np.array(images)
    labels = np.array(label)

    encoder = LabelBinarizer()
    label_vec = encoder.fit_transform(labels)
    
    with open(target_path, 'wb') as file:
        pickle.dump((pixels, label_vec), file)
        file.close()
    
    print('All images are saved.')

def loadData(path):
    with open(path, 'rb') as file:
        # Load images from data file
        (pixels, label_ids) = pickle.load(file)
        file.close()
    print('Check shape of images and label_ids')
    print(pixels.shape)
    print(label_ids.shape)

    return pixels, label_ids

base_path = 'vnts_dataset/datasets/data_img'
data_file_path = 'vnts_dataset/data.data'
target_size = (224, 224)
img_threshold = 200

pixels = []
labels = []
paths = os.listdir(base_path)

# Check number of image in each folder. If it less than img_threshold, create new image by function balance()
print('Check number of image each folder.')
for path in paths:
    print('Process '+path)
    result = balance(base_path+'/'+path, img_threshold)
    print(result)

# Create path for images
image_paths = []
for folder in paths:
    for file in os.listdir(base_path+'/'+folder):
        image_paths.append([base_path+'/'+folder+'/'+file, folder])

random.shuffle(image_paths)

# Read image from image file and write it to data file by function saveData()
print('Read and write image to data file...')
for image in image_paths:
    img = cv2.imread(image[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    pixels.append(img)
    labels.append(image[1])

saveData(data_file_path, pixels, labels)