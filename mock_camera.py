import cv2
import numpy as np
from PIL import Image

class MockCamera:
    def __init__(self, resolution=(1920, 1080), camera_index=0):
        self.camera_index = camera_index
        self.video_capture = cv2.VideoCapture(self.camera_index)
        self.resolution = resolution if resolution is not None else (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.is_running = True
        self.configure()

    def configure(self):
        if not self.video_capture.isOpened():
            raise Exception(f"Could not open webcam at index {self.camera_index}")
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        print(f"Webcam configured to {self.resolution}")

    def start_feed(self):
        pass

    def stop_feed(self):
        self.is_running = False
        print("Webcam preview stopped")

    def capture_frame(self):
        # Capture frame from webcam
        ret, frame = self.video_capture.read()
        if not ret:
            print("Error capturing frame from webcam")
            # Create a dummy image (e.g., a black image) as fallback
            return None
        
        return frame

    def close(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        print("Webcam closed")