import cv2
from decouple import config

# Config
API_KEY = config("ROBOFLOW_API_KEY")
WORKSPACE = "renzotest"
WORKFLOW_ID = "back-squat"

# Knee only state
MIDDLE_STATE = 'middle'

# Squat only states
ASCENDING_STATE = 'ascending'
DESCENDING_STATE = 'descending'
START_STATE = 'start'

# Squat and Knee shared states
UP_STATE = 'up'
DOWN_STATE = 'down'

SQUAT_STATES_COLORS = {
    UP_STATE: (0, 255, 0),
    DOWN_STATE: (0, 255, 255),
    START_STATE: (100, 100, 100),
    DESCENDING_STATE: (0, 127, 127),
    ASCENDING_STATE: (0, 127, 0),
}

# Paths
input_path = "RenzoSquatImg.jpg"  # 👈 replace with your own image
output_path = "output_squat.jpg"

from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # use local inference server
    api_key=API_KEY
)

# Read image and encode to base64
with open(input_path, "rb") as f:
    result = client.run_workflow(
        workspace_name="renzotest",
        workflow_id="back-squat",
        images={
            "webcam": input_path
        }
    )

print("Raw result:", result)

# Parse predictions
data = result[0]

def calculate_squat_state(previous_squat_state, current_knee_state):
    if previous_squat_state == START_STATE and current_knee_state == UP_STATE:
        print(f"Exercise started")
        return UP_STATE, 0
    elif previous_squat_state == UP_STATE and current_knee_state == MIDDLE_STATE:
        print(f"Descending phase")
        return DESCENDING_STATE, 0
    elif previous_squat_state == DESCENDING_STATE and current_knee_state == DOWN_STATE:
        print(f"Descending completed")
        return DOWN_STATE, 0
    elif previous_squat_state == DOWN_STATE and current_knee_state == MIDDLE_STATE:
        print(f"Ascending phase")
        return ASCENDING_STATE, 0
    elif previous_squat_state == ASCENDING_STATE and current_knee_state == UP_STATE:
        print("Ascending completed")
        return UP_STATE, 1
    return previous_squat_state, 0


knee_angle = data.get("knee_angle", -1)
knee_state = data.get("knee_state", MIDDLE_STATE)
squat_state, rep_increment = calculate_squat_state(START_STATE, knee_state)
squat_reps = data.get("squat", {}).get("squat_reps", 0) + rep_increment

print("Knee Angle:", knee_angle)
print("Squat Reps:", squat_reps)
print("Squat State:", squat_state)

# Draw overlay on image
frame = cv2.imread(input_path)
if knee_angle != -1:
    cv2.putText(frame, f"Reps: {squat_reps}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Knee Angle: {int(knee_angle)}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"State: {squat_state}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Draw overlays
    cv2.putText(frame, f"Reps: {squat_reps}",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Knee Angle: {int(knee_angle)}",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2, cv2.LINE_AA)

    # If model has bounding box predictions, draw them

    for box in data.get("object_detection_predictions").get('predictions'):
        x1, y1, width, height = map(int, (box['x'], box['y'], box['width'], box['height']))
        x1 -= width // 2
        x2 = x1 + width
        y1 -= height // 2
        y2 = y1 + height
        label = box.get("class")
        color = SQUAT_STATES_COLORS[squat_state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {squat_state}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)

# Save processed image
cv2.imwrite(output_path, frame)
print(f"✅ Processed image saved at {output_path}")
