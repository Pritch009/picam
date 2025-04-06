import cv2
import numpy as np

class MotionDetector:
    def __init__(self, mode="auto", sensitivity=0.25, min_area=300):
        """
        mode: "auto", "normal", or "lowlight"
        sensitivity: 0 to 1, lower = more sensitive
        min_area: minimum area of motion (in pixels) to count as motion
        """
        self.mode = mode
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.previous_frame = None
        self.motion_detected = False

    def detect_motion(self, frame):
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Resize if needed (optional for lores)
        # gray = cv2.resize(gray, (320, 240))

        # Blur to reduce noise, keep small kernel for low light
        gray = cv2.blur(gray, (3, 3))

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        if self.mode == "auto":
            return self._detect_motion_auto(gray)
        elif self.mode == "normal":
            return self._detect_motion_normal(gray)
        elif self.mode == "lowlight":
            return self._detect_motion_lowlight(gray)
        else:
            raise ValueError("Invalid motion mode")

    def _detect_motion_auto(self, gray):
        # Basic logic to switch based on contrast (e.g., IR mode)
        contrast = gray.std()
        if contrast < 15:
            return self._detect_motion_lowlight(gray)
        else:
            return self._detect_motion_normal(gray)

    def _detect_motion_normal(self, gray):
        delta = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(delta, int(255 * self.sensitivity), 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = any(cv2.contourArea(c) > self.min_area for c in contours)

        self.previous_frame = gray
        return self.motion_detected

    def _detect_motion_lowlight(self, gray):
        delta = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(delta, 8, 255, cv2.THRESH_BINARY)  # Lower threshold for IR
        thresh = cv2.dilate(thresh, None, iterations=1)

        # Add pixel count fallback for IR
        motion_pixels = cv2.countNonZero(thresh)
        self.motion_detected = motion_pixels > (self.min_area * 2)

        self.previous_frame = gray
        return self.motion_detected

    def get_motion_status(self):
        return self.motion_detected

    def reset(self):
        self.previous_frame = None
        self.motion_detected = False
