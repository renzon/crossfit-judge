from ultralytics import YOLO
from pathlib import Path

model = YOLO('yolov8n-pose.pt')

root_dir=Path(__file__).resolve().parent
print(root_dir)

results = model.train(data=root_dir /'glue-stick-runs-dataset-4'/'data.yaml', epochs=100, imgsz=640)