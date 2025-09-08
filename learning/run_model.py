from pathlib import Path

from ultralytics import YOLO
from PIL import Image

root_dir = Path(__file__).resolve().parent

model = YOLO(root_dir / 'Glue-Stick-Project-Runs/runs/pose/train/weights/best.pt')

results = model(root_dir / 'has_glue.jpg')
# results = model('same_from_training.jpg')


r = results[0]
im_array = r.plot()  # plot a BGR numpy array of predictions
im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# Save to file
im.save(root_dir / "second_pose_result.jpg")

print("✅ Pose estimation saved as second_pose_result.jpg")

print(results)
