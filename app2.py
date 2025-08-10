import gradio as gr
import cv2
import os
import json
from datetime import datetime
from yolo_detector import YoloDetector
from openai_client import build_prompt, query_openai
import numpy as np
from PIL import Image
import io
import threading
import queue
import time
from openai_client import build_prompt, query_openai, set_role_description

# Ensure YOLO output directory exists
YOLO_OUTPUT_DIR = "yolo_outputs"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

# Create YOLO detector instance
detector = YoloDetector()

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
        print(f"Error processing image: {e}")
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

def capture_frame_threaded():
    """Capture frame using threading - will fail on macOS Docker but kept for compatibility"""
    frame_queue = queue.Queue()
    
    def capture_worker():
        cap = None
        try:
            # Try different backends
            for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
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
        return None, "Camera not accessible (expected in Docker on macOS)"

def capture_and_analyze():
    """Capture photo with camera and analyze it"""
    frame, message = capture_frame_threaded()
    
    if frame is not None:
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO analysis
        output_image_path, thought, emotion, action = run_yolo_analysis(frame)
        
        if output_image_path and os.path.exists(output_image_path):
            # Load the analyzed image for display
            analyzed_image = cv2.imread(output_image_path)
            analyzed_image_rgb = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB)
            
            # Format the results
            results_text = f"**Thought:** {thought}\n\n**Emotion:** {emotion}\n\n**Action:** {action}"
            
            return analyzed_image_rgb, results_text
        else:
            return frame_rgb, f"Analysis failed: {thought}"
    else:
        # Return a message image instead of blank
        msg_image = np.full((300, 600, 3), 128, dtype=np.uint8)  # Gray background
        error_msg = f"Camera Error: {message}\n\nℹ️ On macOS with Docker, use the Browser Camera or Upload Image instead!"
        return msg_image, error_msg

def analyze_uploaded_image(uploaded_file):
    """Analyze uploaded image file"""
    if uploaded_file is None:
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank_image, "No image uploaded"
    
    try:
        # Process the uploaded image
        frame = process_image_for_yolo(uploaded_file)
        
        if frame is not None:
            output_image_path, thought, emotion, action = run_yolo_analysis(frame)
            
            if output_image_path and os.path.exists(output_image_path):
                # Load the analyzed image for display
                analyzed_image = cv2.imread(output_image_path)
                analyzed_image_rgb = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB)
                
                # Format the results
                results_text = f"**Thought:** {thought}\n\n**Emotion:** {emotion}\n\n**Action:** {action}"
                
                return analyzed_image_rgb, results_text
            else:
                return np.array(uploaded_file), f"Analysis failed: {thought}"
        else:
            return np.array(uploaded_file), "Failed to process the uploaded image"
            
    except Exception as e:
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank_image, f"Error processing image: {str(e)}"

def analyze_webcam_image(webcam_image):
    """Analyze image from browser webcam"""
    if webcam_image is None:
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return None, blank_image, "No webcam image captured", gr.update(visible=False), gr.update(visible=True)
    
    try:
        # Process the webcam image
        frame = process_image_for_yolo(webcam_image)
        
        if frame is not None:
            output_image_path, thought, emotion, action = run_yolo_analysis(frame)
            
            if output_image_path and os.path.exists(output_image_path):
                # Load the analyzed image for display
                analyzed_image = cv2.imread(output_image_path)
                analyzed_image_rgb = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB)
                
                # Format the results
                results_text = f"**Thought:** {thought}\n\n**Emotion:** {emotion}\n\n**Action:** {action}"
                
                # Clear original image, hide analyze button, show take another photo button
                return None, analyzed_image_rgb, results_text, gr.update(visible=False), gr.update(visible=True)
            else:
                return None, np.array(webcam_image), f"Analysis failed: {thought}", gr.update(visible=False), gr.update(visible=True)
        else:
            return None, np.array(webcam_image), "Failed to process the webcam image", gr.update(visible=False), gr.update(visible=True)
            
    except Exception as e:
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return None, blank_image, f"Error processing image: {str(e)}", gr.update(visible=False), gr.update(visible=True)

def reset_camera_interface():
    """Reset the camera interface to take another photo"""
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    return None, blank_image, "", gr.update(visible=True), gr.update(visible=False)

def get_debug_info():
    """Get debug information"""
    backends = []
    if cv2.CAP_V4L2: 
        backends.append("V4L2")
    if cv2.CAP_DSHOW: 
        backends.append("DSHOW")
    if cv2.CAP_GSTREAMER: 
        backends.append("GSTREAMER")
    if cv2.CAP_AVFOUNDATION:
        backends.append("AVFoundation (macOS)")
    
    platform_info = ""
    if os.path.exists('/.dockerenv'):
        platform_info = "Running in Docker container"
    else:
        platform_info = "Running natively"
    
    debug_text = f"""
    Platform: {platform_info}
    OpenCV Version: {cv2.__version__}
    Available backends: {', '.join(backends) if backends else 'None detected'}
    YOLO Output Directory: {YOLO_OUTPUT_DIR}
    Directory exists: {os.path.exists(YOLO_OUTPUT_DIR)}
    
    macOS + Docker Camera Info:
    - Direct camera access typically doesn't work in Docker Desktop on macOS
    - Use Browser Camera or Upload Image features instead
    - These provide full functionality without camera compatibility issues
    """
    return debug_text.strip()

# Create Gradio interface
with gr.Blocks(title="CatGPT", theme=gr.themes.Soft()) as app:
    gr.Markdown("# CatGPT")
    gr.Markdown("The Most Purrfect Cat AI! 🐱")
    
    gr.Markdown("### Character Role")
    role_input = gr.Textbox(
        label="What should the AI act like?",
        placeholder="e.g., a dog exploring a park",
        value="a cat experiencing the world"
    )
    set_role_btn = gr.Button("Set Role")

    def update_role(new_role):
        set_role_description(new_role)
        return f"Role set to: {new_role}"

    role_status = gr.Markdown()

    set_role_btn.click(
        fn=update_role,
        inputs=[role_input],
        outputs=[role_status]
    )

    with gr.Tab("📷 Camera"):
        gr.Markdown("## Camera")
        
        # Camera input at the top
        webcam_input = gr.Image(sources=["webcam"], type="pil", label="Capture Photo")
        
        # Buttons - initially show analyze button, hide take another photo button
        with gr.Row():
            webcam_analyze_btn = gr.Button("Analyze Photo", variant="primary", size="lg", visible=True)
            take_another_btn = gr.Button("Take Another Photo", variant="secondary", size="lg", visible=False)
        
        # Results below
        with gr.Row():
            webcam_output_image = gr.Image(label="Analysis Result", type="numpy")
            webcam_results = gr.Markdown(label="AI Analysis")
        
        # Analyze button click - now includes webcam_input in outputs to clear it
        webcam_analyze_btn.click(
            fn=analyze_webcam_image,
            inputs=[webcam_input],
            outputs=[webcam_input, webcam_output_image, webcam_results, webcam_analyze_btn, take_another_btn]
        )
        
        # Take another photo button click
        take_another_btn.click(
            fn=reset_camera_interface,
            inputs=[],
            outputs=[webcam_input, webcam_output_image, webcam_results, webcam_analyze_btn, take_another_btn]
        )
    
    with gr.Tab("📁 Upload Image"):
        gr.Markdown("## Upload Image")
        
        uploaded_file = gr.Image(label="Choose an image file", type="pil")
        analyze_button = gr.Button("Analyze Uploaded Image", variant="primary")
        
        with gr.Row():
            upload_output_image = gr.Image(label="Image Analysis", type="numpy")
            upload_results = gr.Markdown(label="Analysis Results")
        
        analyze_button.click(
            fn=analyze_uploaded_image,
            inputs=[uploaded_file],
            outputs=[upload_output_image, upload_results]
        )
    
    with gr.Tab("🔧 Debug Information"):
        gr.Markdown("## Debug Information")
        debug_button = gr.Button("Get Debug Info")
        debug_output = gr.Textbox(label="Debug Information", lines=15)
        
        debug_button.click(
            fn=get_debug_info,
            inputs=[],
            outputs=[debug_output]
        )

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True              # Enable debug mode
    )