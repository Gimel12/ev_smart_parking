import cv2

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

if __name__ == "__main__":
    devices = list_video_devices()
    if devices:
        print("Available video capture devices:")
        for device in devices:
            cap = cv2.VideoCapture(device)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            print(f"Device {device}: Resolution {width}x{height}")
    else:
        print("No video capture devices found.")
