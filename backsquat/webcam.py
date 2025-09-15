# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
from decouple import config
import numpy as np

SQUAT_DOWN_ANGLE = 73   # knee angle smaller than this = down
SQUAT_UP_ANGLE = 170    # knee angle larger than this = up

squat_state = 'up'
squat_count = 0
first_execution = True
video_writer = None


def detect_squat(knee_angle, current_state):
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
    """Calculate the angle at point b (in degrees) given points a, b, c"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def my_sink(result, video_frame):
    global squat_state, squat_count, first_execution, video_writer

    frame = video_frame.image.copy()

    if first_execution:
        print("Processing started")
        first_execution = False

    predictions = result.get("keypoint_predictions")
    if predictions and 'person' in predictions.data.get('class_name', []):
        keypoints = get_keypoints_dict(predictions)
        knee_angle = angle_between(
            keypoints['left_hip'],
            keypoints['left_knee'],
            keypoints['left_ankle']
        )
        new_state = detect_squat(knee_angle, squat_state)

        if squat_state == 'down' and new_state == 'up':
            squat_count += 1
            print(f"Rep: {squat_count}")
        elif squat_state == 'up' and new_state == 'down':
            print("Squat Down!")

        squat_state = new_state

        # Draw overlays
        cv2.putText(frame, f"State: {squat_state}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {squat_count}",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # If model has bounding box predictions, draw them
        box_preds = result.get("object_detection_predictions")
        if box_preds:
            for box in box_preds:
                x1, y1, x2, y2 = map(int, box["bbox"])  # depends on workflow output
                label = box.get("class_name", "person")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label}: {squat_state}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Show live video
    # cv2.imshow("Squat Detection", frame)

    # Save video
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frame.shape[:2]
        video_writer = cv2.VideoWriter("output_squat.mp4", fourcc, 30, (w, h))
    video_writer.write(frame)

    # Stop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()


# ----------- MAIN -----------
pipeline = InferencePipeline.init_with_workflow(
    api_key=config("ROBOFLOW_API_KEY"),
    workspace_name="renzotest",
    workflow_id="back-squat",
    video_reference="./PriSquat.mp4",  # input video file
    image_input_name="webcam",
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()

# Cleanup
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
