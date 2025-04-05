import cv2


class MotionDetector:
    def __init__(self, sensitivity=0.5):
        self.sensitivity = sensitivity
        self.previous_frame = None
        self.motion_detected = False

    def detect_motion(self, current_frame):
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray_current
            return False

        frame_delta = cv2.absdiff(self.previous_frame, gray_current)
        thresh = cv2.threshold(frame_delta, int(255 * self.sensitivity), 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = any(cv2.contourArea(c) > 500 for c in contours)

        self.previous_frame = gray_current
        return self.motion_detected

    def get_motion_status(self):
        return self.motion_detected
    
    def reset(self):
        self.previous_frame = None
        self.motion_detected = False