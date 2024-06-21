# Detect and tracking traffic signs on video, recognize the traffic signs and show it by draw the bounding box with their id.

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd

labels_f = pd.read_csv('vnts_dataset/labels2.csv', delimiter=';', header=None)
categories = np.array(labels_f.iloc[:, 0])

save = False
stream = True
conf_threshold = 0.7
target_size = (224, 224)
colors = np.random.randint(0,255, size=(len(categories),3))

font = cv2.FONT_HERSHEY_SIMPLEX

print('Preparing detection model.')
yolo = YOLO('weights/yolov8_trafficsign.pt')
print('Preparing classification model')
model = tf.keras.models.load_model('weights/mobilenet_model.h5')
print('Prepare tracker.')
tracker = DeepSort(max_age=30)
tracks = []

cap = cv2.VideoCapture('vnts_dataset/datasets/data_video/video2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, dsize=None, fx=0.8, fy=0.8)

    detections = yolo(frame, save=save, stream=stream, conf=conf_threshold, imgsz=640)

    detection = []

    for detect in detections:
        for bbox in detect.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, target_size)
            roi = np.array(roi, dtype='float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = tf.keras.applications.mobilenet.preprocess_input(roi)
            predict = model.predict(roi)
            label_id = np.argmax(predict, axis=1)[0]
            label = categories[label_id]

            if label == 'EMPTY':
                continue

            detection.append([ [x1, y1, x2 - x1, y2 - y1], 1, label_id])
        
    tracks = tracker.update_tracks(detection, frame=frame)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            R, G, B = map(int, colors[label_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (R, G, B), 2)
            cv2.putText(frame, categories[label_id], (x1, y1-10), font, 0.5, (R, G, B), 1, cv2.LINE_AA)

    cv2.imshow('Traffic signs recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()