import cv2
from ultralytics import YOLO

image = cv2.imread(r"D:\Coding\train-poon-poon\fruit-detect\fruits.jpg")
model = YOLO(r"D:\Coding\train-poon-poon\fruit-detect\best.pt")

detect = model.predict(image)[0]

CONFIDENCE_LEVEL = 0.5

for d in detect:
    confidence = float(d.boxes.conf)
    class_id = int(d.boxes.cls)
    x1, y1, x2, y2 = [int(x) for x in d.boxes.xyxy[0]]
    print("Confidence:", confidence)
    print("Class ID:", class_id)
    print("Position:", x1, y1, x2, y2)
    print()

    if float(d.boxes.conf) > CONFIDENCE_LEVEL:
        cv2.putText(image, f"{class_id} - {confidence:0.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
