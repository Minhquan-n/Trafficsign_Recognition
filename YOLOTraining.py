from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def trainYOLO():
    print('Load YOLOv8n weight.')
    model = YOLO('weights/yolov8n.pt')

    print('Start training on traffic signs datasets.')
    train = model.train(data='vnts_dataset/datasets/trafficsignyolov8/data.yaml', epochs=60, imgsz=640, device=0)
    print('Save weight of model')
    model.save('weights/yolov8n_custom.pt')

def testYOLOCustom(img):
    yolo = YOLO('weights/yolov8n_custom.pt')

    detections = yolo(img, save=False, stream=True, conf=0.7, imgsz=640, device='cuda:0')

    for detect in detections:
        for bbox in detect.boxes.xyxy:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Traffic Sign', (x1, y1 - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    trainYOLO()

    img = cv2.imread('vnts_dataset/test.jpg')
    testYOLOCustom(img)