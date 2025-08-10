from ultralytics import YOLO
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        """
        Initialize YOLO detector
        
        Args:
            model_path (str): Path to YOLO model file
            conf_threshold (float): Confidence threshold for detections
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            print(f"YOLO model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            list: List of detection dictionaries
        """
        if frame is None or frame.size == 0:
            print("Warning: Empty or None frame provided to detect_objects")
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        try:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            conf = float(box.conf[0])
                            
                            if conf > self.conf_threshold:
                                # Get bounding box coordinates
                                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                                
                                detections.append({
                                    "label": cls_name,
                                    "confidence": conf,
                                    "bbox": bbox,
                                    "class_id": cls_id
                                })
                        except Exception as e:
                            print(f"Error processing detection box: {e}")
                            continue
            
            print(f"Found {len(detections)} objects")
            return detections
            
        except Exception as e:
            print(f"Error in detect_objects: {e}")
            return []
    
    def detect_and_draw(self, frame):
        """
        Detect objects and draw bounding boxes on the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (annotated_frame, detections_list)
        """
        if frame is None or frame.size == 0:
            print("Warning: Empty or None frame provided to detect_and_draw")
            return frame, []
        
        # Get detections
        detections = self.detect_objects(frame)
        
        # Create a copy of the frame for drawing
        boxed_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            try:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = det["label"]
                conf = det["confidence"]
                
                # Ensure coordinates are within frame bounds
                h, w = boxed_frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # Choose color based on class (simple hash-based coloring)
                color_seed = hash(label) % 255
                color = (
                    (color_seed * 50) % 255,
                    (color_seed * 100) % 255, 
                    (color_seed * 150) % 255
                )
                
                # Draw bounding box
                cv2.rectangle(boxed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                text = f"{label} {conf:.2f}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    boxed_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Put label text
                cv2.putText(
                    boxed_frame, 
                    text, 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255),  # White text
                    2
                )
                
            except Exception as e:
                print(f"Error drawing detection {det}: {e}")
                continue
        
        return boxed_frame, detections
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": str(self.model.model),
            "class_names": self.model.names,
            "num_classes": len(self.model.names),
            "confidence_threshold": self.conf_threshold
        }