import cv2
import time
from ultralytics import YOLO
import gradio as gr

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your model
class_labels = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Function to capture and classify an image
def capture_and_classify():
    """
    Captures an image from the webcam and classifies it.
    """
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or could not be opened.")

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture image from webcam.")
    
    # Release the webcam
    cap.release()

    # Perform object detection
    results = model.predict(source=frame, imgsz=640, conf=0.5)

    # Annotate the frame with bounding boxes and labels
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = f"{class_labels[int(cls)]} ({conf:.2f})"
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Gradio Interface
interface = gr.Interface(
    fn=capture_and_classify,
    inputs=None,
    outputs=gr.Image(type="numpy"),
    live=False,
    title="Periodic Waste Classification",
    description="Click the button to capture an image from your webcam and classify it."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
