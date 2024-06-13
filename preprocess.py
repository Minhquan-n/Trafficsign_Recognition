import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelBinarizer

def balance(path):
    images = os.listdir(path)
    count = len(images)
    if count >= 500:
        return 'Done'
    image = 0
    while count < 300:
        img = cv2.imread(path+'/'+images[image])
        k_size = 9
        for _ in range(7):
            gaussian = cv2.GaussianBlur(img, (k_size, k_size), 0)
            cv2.imwrite(path+f'/img_gb{count}.jpg', gaussian)
            count += 1
            k_size += 4
        image += 1

    return 'Done'

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
    pixels = [(np.array(image, dtype=np.float32) / 255.0) for image in pixels]
    pixels = np.array(pixels)
    print('Check shape of images and label_ids')
    print(pixels.shape)
    print(label_ids.shape)

    return pixels, label_ids

base_path = 'vnts_dataset/datasets/data_img'
data_file_path = 'vnts_dataset/data.data'
label_file = 'vnts_dataset/labels2.csv'
target_size = (224, 224)

pixels = []
labels = []
paths = os.listdir(base_path)

print('Check number of image each folder.')
for path in paths:
    print('Process '+path)
    result = balance(base_path+'/'+path)
    print(result)

# Create path for images (max 300 images for each folder. With folder EMPTY is 600 images)
image_paths = []
for folder in paths:
    files = os.listdir(base_path+'/'+folder)[:600] if folder == 'EMPTY' else os.listdir(base_path+'/'+folder)[:300]
    for file in files:
        image_paths.append([base_path+'/'+folder+'/'+file, folder])

random.shuffle(image_paths)

print('Read and write image to data file...')

for image in image_paths:
    img = cv2.imread(image[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    pixels.append(img)
    labels.append(image[1])

saveData(data_file_path, pixels, labels)

print('Check image in data file')
pixels, labels = loadData(data_file_path)

labels_f = pd.read_csv(label_file, delimiter=';', header=None)
categories = np.array(labels_f.iloc[:, 0])

plt.subplots(3, 4)
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(pixels[i])
    plt.axis('off')
    plt.title(categories[np.argmax(labels[i])])
plt.show()