import cv2
import requests
import json
from ultralytics import YOLO
import openai
import os

# --- YOLOv8 model load ---
model = YOLO('yolov8n.pt')  # lightweight YOLOv8 nano model

# --- OpenAI API config ---
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # or any supported model you want

# --- System prompt ---
SYSTEM_PROMPT = """
You are a cat experiencing the world. 
Always respond ONLY with a valid JSON object with keys: "thought", "action", and "emotion".
Respond ONLY with JSON — no extra text or explanation.
Example:
{
  "thought": "Enemy vacuum spotted. Must proceed with caution.",
  "action": "hide",
  "emotion": "anxious"
}
"""

# --- Helper: Run YOLO inference on a frame ---
def detect_objects(frame):
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.3:
                detections.append({"label": cls, "confidence": conf})
    return detections

# --- Helper: Build prompt for LLM ---
def build_prompt(detections):
    objects = [d["label"] for d in detections]
    if not objects:
        objects_desc = "nothing notable"
    else:
        objects_desc = ", ".join(objects)
    user_prompt = (
        f"I see the following objects around me: {objects_desc}. "
        f"I am a cat. Based on this, what are my thought, action, and emotion?"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

# --- Helper: Query OpenAI API ---
def query_openai(messages):
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_completion_tokens": 400,
        "stream": False,
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# --- Main loop: webcam + detection + LLM ---
def run_webcam_pipeline():
    cap = cv2.VideoCapture(0)  # open default webcam
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Run detection
        detections = detect_objects(frame)
        print(f"🦴 YOLO detected objects: {[d['label'] for d in detections]}")

        # Build prompt and query LLM
        prompt = build_prompt(detections)
        print(f"📤 Prompt sent to OpenAI:\n{json.dumps(prompt, indent=2)}")

        try:
            response_text = query_openai(prompt)
            print(f"😺 Raw LLM response:\n{response_text}")

            try:
                cat_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"❗ JSON parse error: {e}")
                cat_response = {"raw_response": response_text}
            print(f"Parsed response:\n{json.dumps(cat_response, indent=2)}")

        except Exception as e:
            print(f"❌ OpenAI API error: {e}")

        # Optional: show webcam
        cv2.imshow("Webcam", frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_pipeline()
