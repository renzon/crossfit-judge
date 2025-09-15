# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
from decouple import config

import numpy as np

SQUAT_DOWN_ANGLE = 73  # knee angle smaller than this = down
SQUAT_UP_ANGLE = 170    # knee angle larger than this = up

squat_state = 'up'
squat_count = 0
first_execution = True

def detect_squat(knee_angle, current_state):
    # print(f'knee_angle: {knee_angle} ')
    if knee_angle < SQUAT_DOWN_ANGLE and current_state != 'down':
        return 'down'
    elif knee_angle > SQUAT_UP_ANGLE and current_state == 'down':
        return 'up'
    return current_state

def get_keypoints_dict(predictions):
    names = predictions.data['keypoints_class_name'][0]
    coords = predictions.data['keypoints_xy'][0]
    return {name: np.array(coord) for name, coord in zip(names, coords)}

def angle_between(a, b, c):
    """
    Calculate the angle at point b (in degrees) given points a, b, c
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def my_sink(result, video_frame):
    global squat_state
    global squat_count
    global first_execution
    if first_execution:
        print("Processing started")
        first_execution = False
    predictions = result.get("keypoint_predictions")
    if not predictions or 'person' not in predictions.data.get('class_name', []):
        return

    keypoints = get_keypoints_dict(predictions)
    knee_angle = angle_between(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
    new_state = detect_squat(knee_angle, squat_state)

    if squat_state == 'down' and new_state == 'up':
        squat_count += 1
        print(f"Rep: {squat_count}")
    elif squat_state == 'up' and new_state == 'down':
        print("Squat Down!")


    squat_state = new_state


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=config("ROBOFLOW_API_KEY"),
    workspace_name="renzotest",
    workflow_id="back-squat",
    video_reference="./RenzoSquat.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    # video_reference="./PriSquat.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    image_input_name="webcam",
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish