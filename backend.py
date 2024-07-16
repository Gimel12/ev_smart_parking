import cv2
import numpy as np
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

# Fixed ROI parameters
ROI_X = 255
ROI_Y = 287
ROI_WIDTH = 146
ROI_HEIGHT = 86

latest_results = {"timestamp": None, "cars_detected": 0, "spots_available": 2, "image_with_boxes": ""}

# Global variables
selected_device = None
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

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

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase brightness and contrast
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Histogram equalization
    equalized = cv2.equalizeHist(adjusted)
    
    return equalized

def detect_cars(image):
    # Apply background subtraction
    fg_mask = background_subtractor.apply(image)
    
    # Threshold the image
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area = 500  # Adjust this value based on your specific scenario
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return len(large_contours), large_contours

def capture_and_analyze():
    global latest_results, selected_device
    
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
    
    # Enhance the image
    enhanced_roi = enhance_image(roi)
    
    # Detect cars
    cars_detected, contours = detect_cars(enhanced_roi)
    
    # Calculate spots available (assuming 2 total spots)
    spots_available = max(0, 2 - cars_detected)
    
    # Draw contours on the original ROI
    result_image = roi.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # Convert the image to base64 for sending to frontend
    _, buffer = cv2.imencode('.png', result_image)
    img_str = base64.b64encode(buffer).decode()
    
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
    
    # Schedule the task to run every 5 minutes
    schedule.every(5).minutes.do(capture_and_analyze)
    
    # Run the scheduled task immediately once
    capture_and_analyze()
    
    # Run the schedule in a separate thread
    threading.Thread(target=run_schedule, daemon=True).start()
    
    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
