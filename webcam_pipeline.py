# webcam_pipeline.py
import cv2
import json
import time

from yolo_detector import YoloDetector
from openai_client import build_prompt, query_openai

REQUESTS_PER_SECOND = 1
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND

def run_webcam_pipeline():
    detector = YoloDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🎥 Webcam started. Press 'q' to quit.")
    last_request_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        current_time = time.time()
        elapsed = current_time - last_request_time

        # Detect objects
        detections = detector.detect_objects(frame)
        print(f"🦴 YOLO detected objects: {[d['label'] for d in detections]}")

        if elapsed >= REQUEST_DELAY:
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

            last_request_time = current_time
        else:
            print(f"⏳ Waiting to send next API request... ({REQUEST_DELAY - elapsed:.2f}s left)")

        # Show webcam feed
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_pipeline()
