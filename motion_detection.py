import cv2
import cv2
import numpy as np

class MotionDetector:
    def __init__(self, sensitivity=0.2, min_area=500):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.previous_frame = None
        self.motion_detected = False
        self.alpha = 0.5  # Weight for accumulateWeighted

    def detect_motion(self, current_frame):
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Fast box blur instead of expensive Gaussian
        gray = cv2.blur(gray, (9, 9))

        # Initialize background frame
        if self.previous_frame is None:
            self.previous_frame = gray.astype("float")
            return False

        # Update running average background
        cv2.accumulateWeighted(gray, self.previous_frame, self.alpha)

        # Compute difference between background and current frame
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.previous_frame))

        # Threshold the delta image
        _, thresh = cv2.threshold(frame_delta, int(255 * self.sensitivity), 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = any(cv2.contourArea(c) > self.min_area for c in contours)

        return self.motion_detected

    def get_motion_status(self):
        return self.motion_detected

    def reset(self):
        self.previous_frame = None
        self.motion_detected = False
