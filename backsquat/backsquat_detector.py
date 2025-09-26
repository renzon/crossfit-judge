# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
from decouple import config

SQUAT_DOWN_ANGLE = 73  # knee angle smaller than this = down
SQUAT_UP_ANGLE = 170  # knee angle larger than this = up

# Knee only state
MIDDLE_STATE = 'middle'

# Squat only states
ASCENDING_STATE = 'ascending'
DESCENDING_STATE = 'descending'
START_STATE = 'start'

# Squat and Knee shared states
UP_STATE = 'up'
DOWN_STATE = 'down'

SQUAT_STATES_COLORS={
    UP_STATE: (0, 255, 0),
    DOWN_STATE: (0, 255, 255),
    START_STATE: (100, 100, 100),
    DESCENDING_STATE: (0, 127, 127),
    ASCENDING_STATE: (0, 127, 0),
}

# State Variables
squat_state = START_STATE
squat_reps = 0
video_writer = None


def detect_knee_state(knee_angle):
    if knee_angle < SQUAT_DOWN_ANGLE:
        return DOWN_STATE
    elif knee_angle > SQUAT_UP_ANGLE:
        return UP_STATE
    return MIDDLE_STATE


def calculate_squat_state(knee_state):
    global squat_state, squat_reps
    if squat_state == START_STATE and knee_state == UP_STATE:
        print(f"Exercise started")
        squat_state = UP_STATE
    elif squat_state == UP_STATE and knee_state == MIDDLE_STATE:
        squat_state = DESCENDING_STATE
        print(f"Descending phase")
    elif squat_state == DESCENDING_STATE and knee_state == DOWN_STATE:
        squat_state = DOWN_STATE
        print(f"Descending completed")
    elif squat_state == DOWN_STATE and knee_state == MIDDLE_STATE:
        squat_state = ASCENDING_STATE
        print(f"Ascending phase")
    elif squat_state == ASCENDING_STATE and knee_state == UP_STATE:
        squat_state = UP_STATE
        squat_reps += 1
        print(f"Rep: {squat_reps}")
        print("Ascending completed")
    return squat_state





def my_sink(result, video_frame):
    global video_writer

    frame = video_frame.image.copy()

    knee_angle = result.get("knee_angle")
    if knee_angle is not None:
        knee_state = detect_knee_state(knee_angle)
        calculate_squat_state(knee_state)

        # Draw overlays
        cv2.putText(frame, f"Reps: {squat_reps}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)

        # If model has bounding box predictions, draw them
        box_preds = result.get("object_detection_predictions")
        if box_preds:
            for box in box_preds:
                x1, y1, x2, y2 = map(int, box[0])  # depends on workflow output
                label = box[5].get("class_name")
                color = SQUAT_STATES_COLORS[squat_state]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {squat_state}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2, cv2.LINE_AA)

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
    workflow_id="person-with-keypoints",
    video_reference="./RenzoSquat.mp4",  # input video file
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
