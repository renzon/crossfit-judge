from pathlib import Path

from ultralytics import YOLO
from PIL import Image

root_dir = Path(__file__).resolve().parent

model = YOLO(root_dir / 'glue-stick-runs/runs/pose/train3/weights/best.pt')

results = model(root_dir / 'has_glue.jpg')
# results = model('same_from_training.jpg')


r = results[0]
im_array = r.plot()  # plot a BGR numpy array of predictions
im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# Save to file
im.save(root_dir / "pose_result.jpg")

print("✅ Pose estimation saved as pose_result.jpg")

print(results)
