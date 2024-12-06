import cv2
import time
import threading
from ultralytics import YOLO
import gradio as gr

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your model
class_labels = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Initialize global variables
cap = cv2.VideoCapture(0)
latest_frame = None
stop_thread = False
lock = threading.Lock()  # To synchronize access to `latest_frame`

def capture_frames():
    """
    Continuously captures frames from the webcam.
    """
    global cap, latest_frame, stop_thread, lock
    while not stop_thread:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
        time.sleep(0.1)  # Capture every 100ms for smooth real-time view

def get_real_time_view():
    """
    Returns the latest frame for real-time view.
    """
    global latest_frame, lock
    with lock:
        if latest_frame is None:
            return None
        frame = latest_frame.copy()

    # Convert BGR to RGB for display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def process_frame():
    """
    Processes the latest captured frame and adds bounding boxes.
    """
    global latest_frame, lock
    with lock:
        if latest_frame is None:
            return None
        frame = latest_frame.copy()

    # Perform object detection
    results = model.predict(source=frame, imgsz=640, conf=0.5)

    # Annotate the frame with bounding boxes and labels
    if results[0].boxes.data is not None:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            label = f"{class_labels[int(cls)]} ({conf:.2f})"
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for Gradio
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Start the frame capture thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

# Gradio Interface
interface = gr.Interface(
    fn=lambda: (get_real_time_view(), process_frame()),
    inputs=None,
    outputs=[
        gr.Image(type="numpy", label="Real-Time Camera View"),
        gr.Image(type="numpy", label="Processed Detection View"),
    ],
    live=True,  # Automatically update both views
    title="Real-Time Waste Classification",
    description="Left: Real-time camera view. Right: Processed frames with bounding boxes updated every 5 seconds."
)

# Ensure the webcam is released when the script exits
import atexit
@atexit.register
def cleanup():
    global stop_thread, cap
    stop_thread = True
    if cap.isOpened():
        cap.release()
    print("Webcam released and resources cleaned up.")

if __name__ == "__main__":
    interface.launch()
