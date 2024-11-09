import cv2
from ultralytics import YOLO

image = cv2.imread(r"D:\Coding\train-poon-poon\fruit-detect\fruits.jpg")
model = YOLO(r"D:\Coding\train-poon-poon\fruit-detect\best.pt")

detect = model.predict(image)[0]

cv2.imshow("Result", detect.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
