import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import time
import schedule
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8n.pt')

# Global variables for ROI
ROI_X = 400
ROI_Y = 300
ROI_WIDTH = 300
ROI_HEIGHT = 200

latest_results = {"timestamp": None, "cars_detected": 0, "spots_available": 2, "image_with_boxes": ""}

# Global variable to store the selected device
selected_device = None

def list_video_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def select_video_device():
    global selected_device
    devices = list_video_devices()
    if not devices:
        raise Exception("No video capture devices found.")
    elif len(devices) == 1:
        selected_device = devices[0]
    else:
        print("Available video capture devices:")
        for i, device in enumerate(devices):
            cap = cv2.VideoCapture(device)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            print(f"{i}: Device {device} (Resolution: {width}x{height})")
        choice = int(input("Select a device by number: "))
        selected_device = devices[choice]

def adjust_roi(frame):
    global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT

    def update_roi(*args):
        global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        ROI_X = cv2.getTrackbarPos('X', 'Adjust ROI')
        ROI_Y = cv2.getTrackbarPos('Y', 'Adjust ROI')
        ROI_WIDTH = cv2.getTrackbarPos('Width', 'Adjust ROI')
        ROI_HEIGHT = cv2.getTrackbarPos('Height', 'Adjust ROI')
        
        # Ensure ROI stays within image boundaries
        h, w = frame.shape[:2]
        ROI_X = min(ROI_X, w - ROI_WIDTH)
        ROI_Y = min(ROI_Y, h - ROI_HEIGHT)
        
        temp = frame.copy()
        cv2.rectangle(temp, (ROI_X, ROI_Y), (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 0), 2)
        cv2.imshow('Adjust ROI', temp)

    cv2.namedWindow('Adjust ROI')
    cv2.createTrackbar('X', 'Adjust ROI', ROI_X, frame.shape[1], update_roi)
    cv2.createTrackbar('Y', 'Adjust ROI', ROI_Y, frame.shape[0], update_roi)
    cv2.createTrackbar('Width', 'Adjust ROI', ROI_WIDTH, frame.shape[1], update_roi)
    cv2.createTrackbar('Height', 'Adjust ROI', ROI_HEIGHT, frame.shape[0], update_roi)

    update_roi()  # Initial call to draw rectangle

    print("Adjust the ROI using the trackbars. Press 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Final ROI: X={ROI_X}, Y={ROI_Y}, Width={ROI_WIDTH}, Height={ROI_HEIGHT}")

def capture_and_analyze():
    global latest_results, selected_device, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
    
    if selected_device is None:
        select_video_device()
    
    # Capture image from selected webcam
    cap = cv2.VideoCapture(selected_device)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to capture image from webcam (device {selected_device})")
        return
    
    # Crop the image to the region of interest
    roi = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
    
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    
    # Perform detection
    results = model(image)
    
    cars_detected = 0
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'car' and box.conf.item() >= 0.25:
                cars_detected += 1
    
    # Calculate spots available (assuming 2 total spots)
    spots_available = max(0, 2 - cars_detected)
    
    # Draw bounding boxes on the image
    img = results[0].plot()
    
    # Convert the image to base64 for sending to frontend
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Update latest results
    latest_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cars_detected": cars_detected,
        "spots_available": spots_available,
        "image_with_boxes": img_str
    }
    
    print(f"Analysis complete: {cars_detected} cars detected, {spots_available} spots available")

@app.get("/latest_results")
async def get_latest_results():
    return JSONResponse(content=latest_results)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    select_video_device()  # Select device at startup
    
    # Capture initial frame for ROI adjustment
    cap = cv2.VideoCapture(selected_device)
    ret, initial_frame = cap.read()
    cap.release()
    
    if ret:
        adjust_roi(initial_frame)
    else:
        print("Failed to capture initial frame for ROI adjustment")
        exit(1)
    
    # Schedule the task to run every 5 minutes
    schedule.every(5).minutes.do(capture_and_analyze)
    
    # Run the scheduled task immediately once
    capture_and_analyze()
    
    # Run the schedule in a separate thread
    threading.Thread(target=run_schedule, daemon=True).start()
    
    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
