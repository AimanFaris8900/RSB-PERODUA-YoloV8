from ultralytics import YOLO
from ultralytics.data.split import autosplit

# autosplit(
#     path='datasets/images', 
#     weights=(0.8, 0.1, 0.1), # (train, val, test) ratios
#     annotated_only=False     # if True, only split images with existing labels
# )

model = YOLO("yolo26n-seg.pt")
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)

print(results)