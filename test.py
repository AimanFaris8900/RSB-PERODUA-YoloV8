from ultralytics import YOLO
import cv2

model = YOLO("runs/segment/train-4/weights/best.pt")

result = model.predict(source="test_images/test_4.jpg", show=True, save=True)

# def camera_feed():
#     cam = cv2.VideoCapture()

if __name__ == "__main__":
    pass