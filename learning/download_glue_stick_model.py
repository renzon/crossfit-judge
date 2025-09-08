from roboflow import Roboflow
from decouple import config

rf = Roboflow(api_key=config("ROBOFLOW_API_KEY"))
project = rf.workspace("renzotest").project("detect-squat-1egea")
dataset = project.version(1).download("yolov5")