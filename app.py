import streamlit as st
import cv2
import os
import json
from datetime import datetime
from yolo_detector import YoloDetector
from openai_client import build_prompt, query_openai
import numpy as np
from PIL import Image
import io

# Ensure YOLO output directory exists
YOLO_OUTPUT_DIR = "yolo_outputs"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

# Create YOLO detector instance
@st.cache_resource
def get_detector():
    """Cache the detector to avoid reloading the model"""
    return YoloDetector()

detector = get_detector()

def process_image_for_yolo(image_data):
    """Process image data for YOLO detection"""
    try:
        # Convert PIL image to numpy array
        if hasattr(image_data, 'mode'):
            # PIL Image
            image_array = np.array(image_data)
        else:
            # Assume it's already a numpy array
            image_array = image_data
            
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            frame = image_array
            
        return frame
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def run_yolo_analysis(frame):
    """Run YOLO detection and OpenAI reasoning on a frame"""
    if frame is None:
        return None, "No frame provided", "", ""
    
    try:
        # Run detection and draw boxes
        boxed_frame, detections = detector.detect_and_draw(frame)
        
        # Save the image with bounding boxes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = os.path.join(YOLO_OUTPUT_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(output_image_path, boxed_frame)
        
        # Prepare prompt for OpenAI
        prompt = build_prompt(detections)
        
        # Query OpenAI and parse the JSON response
        response_text = query_openai(prompt)
        thought = emotion = action = ""
        
        try:
            parsed = json.loads(response_text)
            thought = parsed.get("thought", "")
            emotion = parsed.get("emotion", "")
            action = parsed.get("action", "")
        except json.JSONDecodeError as e:
            thought = f"Parsing error from LLM response: {str(e)}"
            emotion = "confused"
            action = "retry"
        
        return output_image_path, thought, emotion, action
        
    except Exception as e:
        return None, f"Error in detection pipeline: {str(e)}", "", ""

# ----------------- STREAMLIT APP -----------------
st.title("CatGPT")

st.subheader("Camera Capture")

def capture_frame_threaded():
    import threading
    import queue
    import time
    
    frame_queue = queue.Queue()
    
    def capture_worker():
        cap = None
        try:
            # Try different backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    # Configure camera
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Give camera time to initialize
                    time.sleep(1)
                    
                    # Read several frames to flush buffer
                    for _ in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            if frame.mean() > 5:  # Not a black frame
                                frame_queue.put(frame)
                                break
                    break
                else:
                    if cap:
                        cap.release()
        except Exception as e:
            frame_queue.put(f"Error: {e}")
        finally:
            if cap:
                cap.release()
    
    # Start capture thread
    thread = threading.Thread(target=capture_worker)
    thread.start()
    thread.join(timeout=10)  # 10 second timeout
    
    if not frame_queue.empty():
        result = frame_queue.get()
        if isinstance(result, str):
            return None, result
        return result, "Success"
    else:
        return None, "Timeout capturing frame"

if st.button("Capture a photo with camera"):
    with st.spinner("Taking photo..."):
        frame, message = capture_frame_threaded()
        
        if frame is not None:
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with st.spinner("Running analysis..."):
                output_image_path, thought, emotion, action = run_yolo_analysis(frame)
                
                if output_image_path and os.path.exists(output_image_path):
                    st.image(output_image_path, caption="Analyzed Photo", use_container_width=True)
                    st.markdown(f"**Thought:** {thought}")
                    st.markdown(f"**Emotion:** {emotion}")
                    st.markdown(f"**Action:** {action}")
                else:
                    st.error(f"Analysis failed: {thought}")
        else:
            st.error(f"Failed to capture frame: {message}")

st.divider()

st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Uploaded Image"):
        with st.spinner("Analyzing uploaded image..."):
            frame = process_image_for_yolo(image)
            
            if frame is not None:
                output_image_path, thought, emotion, action = run_yolo_analysis(frame)
                
                if output_image_path and os.path.exists(output_image_path):
                    st.image(output_image_path, caption="YOLO Detection Output", use_container_width=True)
                    st.markdown(f"**Thought:** {thought}")
                    st.markdown(f"**Emotion:** {emotion}")
                    st.markdown(f"**Action:** {action}")
                else:
                    st.error(f"Analysis failed: {thought}")
            else:
                st.error("Failed to process the uploaded image")

# Debug section
with st.expander("Debug Information"):
    st.write(f"OpenCV Version: {cv2.__version__}")
    st.write(f"Available backends:")
    backends = []
    if cv2.CAP_V4L2: backends.append("V4L2")
    if cv2.CAP_DSHOW: backends.append("DSHOW") 
    if cv2.CAP_GSTREAMER: backends.append("GSTREAMER")
    st.write(backends)