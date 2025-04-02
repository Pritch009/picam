import datetime
import time
from pi_camera import Camera
from motion_detection import MotionDetector
from animal_recognition import AnimalRecognizer
import cv2

from video_database import VideoDatabase

class RichCamera:
    def __init__(
        self,
        model_path="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1",
        video_folder="videos",
        keywords=["man"],
        threshold=0.3,
        recording_duration=300,
        motion_timeout=15
    ):
        # Components
        self.camera = Camera()
        self.motion_detector = MotionDetector()
        self.animal_recognizer = AnimalRecognizer(model_path=model_path, keywords=keywords, threshold=threshold)
        self.video_database = VideoDatabase()
        # Parameters
        self.recording_duration = motion_timeout  # seconds
        self.motion_timeout = recording_duration  # seconds without motion to stop recording
        self.video_database = video_folder
        # State
        self.recording = False
        self.video_writer = None
        self.last_motion_time = None  # Track the last time motion was detected
        self.animals_seen = set()  # Track unique animals seen

    def run_in_background(self):
        start_time = None
        last_motion_time = None
        video_id = None
        frame_errors = 0
        while True:
            # Capture frame
            frame = self.camera.capture_frame()
            current_time = time.time()

            # Check if the frame is valid
            if not frame:
                frame_errors += 1
                print("Error capturing frame")
                time.sleep(1)
                if frame_errors > 5:
                    print("Too many frame errors, stopping...")
                    break
            elif frame_errors > 0:
                frame_errors = 0

            # Detect motion in the frame
            motion_detected = self.motion_detector.get_motion_status()
            if motion_detected:
                last_motion_time = current_time  # Update last motion time
                print("Motion detected, checking for animals...")

            # Detect animals in the frame
            animals = self.animal_recognizer.recognize_animal(frame)
            if animals:
                new_animal = False
                # Draw bounding boxes around recognized animals
                for animal in animals:
                    if animal not in self.animals_seen:
                        new_animal = True
                    self.animals_seen.add(animal[0])  # Add animal class name to seen set
                    frame = self.animal_recognizer.draw_bounding_box(frame, animal)

                if not recording:
                    print("Animals detected, starting recording...")
                    recording = True
                    start_time = current_time

                    # Define video codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{"{self.video_folder}/" if self.video_folder else ""}animal_recording_{timestamp}.mp4"
                    self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

                    # Insert video entry into the database
                    video_id = self.video_database.insert_video(video_filename, start_time, self.animals_seen)
                elif new_animal:
                    # Update the database with new animals
                    self.video_database.update_video_animals(video_id, self.animals_seen)

            # Write the frame to the video file if recording
            if recording:
                # Record frame
                self.video_writer.write(frame)

                # Check for stop conditions
                elapsed_time = current_time - start_time
                motion_since_start = current_time - last_motion_time
                self.video_database.update_video_duration(video_id, elapsed_time)

                # Explicit stop condition
                should_stop = elapsed_time >= self.recording_duration or motion_since_start >= self.motion_timeout

                if should_stop:
                    print("Recording duration or motion timeout reached, stopping recording...")
                    # Update class state
                    self.recording = False
                    self.video_writer.release()
                    self.video_writer = None
                    self.animals_seen.clear()

                    # Update local state
                    motion_detected = False
                    start_time = None
                    last_motion_time = None
                    video_filename = None

            else:
                # Reset motion_detected if no motion is detected for a while
                time.sleep(1)  # Check every 1 second
