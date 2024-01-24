from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

#cap = cv2.VideoCapture("D:\VSC code\OpenCV\Vid\car.mp4") #For video
cap = cv2.VideoCapture(0)
cap.set(0, 384)
cap.set(0, 640)

model = YOLO("D:\\VSC code\\OpenCV\\YOLO-weights\\yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball clove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", " orange", "brocolli", 
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cellphone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #Pretrained objects 


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
        #Bounding box
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            w, h = x2-x1, y2-y1

            #confidence 
            conf = math.ceil((box.conf[0]*100))/100

            #Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass =="car" or currentClass == "motorbike" or currentClass == "bus" or currentClass == "truck" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1)), scale=0.6, thickness=2, offset=3)
                currentArray = np.array([x1,y2,x2,y2,conf])
                detections = np.vstack((detections, currentArray))
            
                
    cv2.imshow("Result", img)
    cv2.waitKey(1)