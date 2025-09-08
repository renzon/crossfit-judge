from roboflow import Roboflow
from decouple import config

rf = Roboflow(api_key=config("ROBOFLOW_API_KEY"))
project = rf.workspace("renzotest").project("glue-stick-project-uilei")
dataset = project.version(5).download("yolov5")