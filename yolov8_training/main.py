from ultralytics import YOLO
from ultralytics.data.split import autosplit



def data_autosplit():
    autosplit(
        path='datasets/images', 
        weights=(0.8, 0.1, 0.1), # (train, val, test) ratios
        annotated_only=False     # if True, only split images with existing labels
    )

def train_yolov8():
    data_autosplit()
    
    model = YOLO("yolo26n-seg.pt")
    results = model.train(data="dataset.yaml", epochs=100, imgsz=640)

    print(results)

if __name__ == "__main__":
    train_yolov8()