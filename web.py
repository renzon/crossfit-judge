from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import base64
from inference_sdk import InferenceHTTPClient
from decouple import config

app = Flask(__name__)

# ---- Config ----
API_KEY = config("ROBOFLOW_API_KEY")
WORKSPACE = "renzotest"
WORKFLOW_ID = "back-squat"

# Initialize Roboflow HTTP client
client = InferenceHTTPClient(
    api_url=config("INFERENCE_SERVER_URL"), # use local inference server
    api_key=API_KEY
)


# ---- States ----
MIDDLE_STATE = 'middle'
ASCENDING_STATE = 'ascending'
DESCENDING_STATE = 'descending'
START_STATE = 'start'
UP_STATE = 'up'
DOWN_STATE = 'down'

SQUAT_STATES_COLORS = {
    UP_STATE: (0, 255, 0),
    DOWN_STATE: (0, 255, 255),
    START_STATE: (100, 100, 100),
    DESCENDING_STATE: (0, 127, 127),
    ASCENDING_STATE: (0, 127, 0),
}

def calculate_squat_state(previous_squat_state, current_knee_state):
    """FSM for squat reps"""
    if previous_squat_state == START_STATE and current_knee_state == UP_STATE:
        return UP_STATE, 0
    elif previous_squat_state == UP_STATE and current_knee_state == MIDDLE_STATE:
        return DESCENDING_STATE, 0
    elif previous_squat_state == DESCENDING_STATE and current_knee_state == DOWN_STATE:
        return DOWN_STATE, 0
    elif previous_squat_state == DOWN_STATE and current_knee_state == MIDDLE_STATE:
        return ASCENDING_STATE, 0
    elif previous_squat_state == ASCENDING_STATE and current_knee_state == UP_STATE:
        return UP_STATE, 1
    return previous_squat_state, 0


@app.route("/")
def index():
    return  render_template_string(HTML_PAGE)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json
    frame_b64 = data["frame"]
    prev_squat_state = data.get("squat_state", START_STATE)
    prev_reps = int(data.get("squat_reps", 0))

    # Decode Base64 to image
    img_bytes = base64.b64decode(frame_b64.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run workflow
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={"webcam": frame}
    )
    data_result = result[0]

    knee_angle = data_result.get("knee_angle", -1)
    knee_state = data_result.get("knee_state", MIDDLE_STATE)

    squat_state, rep_increment = calculate_squat_state(prev_squat_state, knee_state)
    squat_reps = prev_reps + rep_increment

    # Overlay info
    if knee_angle != -1:
        cv2.putText(frame, f"Reps: {squat_reps}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"State: {squat_state}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        for box in data_result.get("object_detection_predictions", {}).get("predictions", []):
            x1, y1, w_box, h_box = map(int, (box['x'], box['y'], box['width'], box['height']))
            x1 -= w_box // 2
            y1 -= h_box // 2
            x2 = x1 + w_box
            y2 = y1 + h_box
            label = box.get("class")
            color = SQUAT_STATES_COLORS[squat_state]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {squat_state}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Encode image back to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_b64 = base64.b64encode(buffer).decode("utf-8")
    processed_frame = f"data:image/jpeg;base64,{processed_b64}"

    return jsonify({
        "frame": processed_frame,
        "squat_state": squat_state,
        "squat_reps": squat_reps,
        "knee_angle": knee_angle
    })

HTML_PAGE="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Live Squat Detection</title>
</head>
<body>
  <h2>Webcam Squat Detection</h2>
  <video id="webcam" autoplay playsinline width="480"></video>
  <canvas id="canvas" style="display:none;"></canvas>

  <h3>Processed Stream</h3>
  <img id="processed" width="480"/>
  <p><b>Reps:</b> <span id="reps">0</span></p>
  <p><b>State:</b> <span id="state">-</span></p>
  <p><b>Knee Angle:</b> <span id="angle">-</span></p>

  <script>
  let squat_state = "start";
  let squat_reps = 0;

  const video = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  // Allowed transitions (FSM)
  const VALID_TRANSITIONS = {
    "start": ["start", "up"],
    "up": ["up", "descending"],
    "descending": ["descending", "down"],
    "down": ["down", "ascending"],
    "ascending": ["ascending", "up"]
  };

  async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  }

  async function sendFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg");

    try {
      const response = await fetch("/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: dataUrl, squat_state, squat_reps })
      });

      const result = await response.json();

      // Only update if transition is valid
      // Need to validate because http responses can be get in a different order than requests were sent
      if (VALID_TRANSITIONS[squat_state].includes(result.squat_state)) {
        squat_state = result.squat_state;
        squat_reps = result.squat_reps; // trust server only if state transition is valid
        document.getElementById("processed").src = result.frame;
        document.getElementById("reps").innerText = squat_reps;
        document.getElementById("state").innerText = squat_state;
        document.getElementById("angle").innerText = result.knee_angle;
      }

    } catch (err) {
      console.error("Frame processing failed:", err);
    }
  }

  setupWebcam().then(() => {
    setInterval(sendFrame, 200); // ~5 FPS
  });
</script>
"""



if __name__ == "__main__":
    app.run(debug=True)


