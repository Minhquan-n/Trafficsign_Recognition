# Preparing YOLO dataset without labeling. Use this dataset for annotating and create dataset for training YOLO.

import cv2
import numpy as np
from ultralytics import YOLO

save = False
stream = True
conf_threshold = 0.2
target_size = (256, 256)

video = ['vnts_dataset/datasets/data_video/0525.mp4', 'vnts_dataset/datasets/data_video/0525(1).mp4']

yolo = YOLO('weights/yolov8_trafficsign.pt')

target_path = 'vnts_dataset/datasets/data_raw/img/vd'

for i, path in enumerate(video):
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
        
        detections = yolo(frame, save=save, stream=stream, conf=conf_threshold, imgsz=640)

        for detect in detections:
            for bbox in detect.boxes.xyxy:
                x1, y1, x2, y2 = np.array(bbox, dtype=int)
                
                roi = frame[y1:y2, x1:x2]
                roi = cv2.resize(roi, target_size)
                file_path = target_path+f'{i}_{n}.jpg'
                print(file_path)
                n += 1
                cv2.imwrite(filename=file_path, img=roi)

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, 'Traffic sign', (x1, y1 - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()