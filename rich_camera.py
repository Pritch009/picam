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
        database_path="video_database.db",
        keywords=["man"],
        threshold=0.3,
        recording_duration=300,
        motion_timeout=15,
        target_framerate=30.0,
    ):
        # Components
        self.camera = Camera()
        self.motion_detector = MotionDetector()
        self.animal_recognizer = AnimalRecognizer(model_path=model_path, keywords=keywords, threshold=threshold)
        # Parameters
        self.recording_duration = recording_duration  # seconds
        self.motion_timeout = motion_timeout  # seconds without motion to stop recording
        self.video_folder = video_folder # Folder to save videos
        self.database_path =  database_path # Path to the SQLite database file
        self.target_framerate = target_framerate  # Target framerate for video recording
        # State
        self.recording = False
        self.video_writer = None
        self.last_motion_time = None  # Track the last time motion was detected
        self.animals_seen = set()  # Track unique animals seen

    def run_in_background(self):
        print("Starting camera...")
        start_time = None
        last_motion_time = time.time()
        video_id = None
        frame_errors = 0
        video_database = VideoDatabase(db_name=self.database_path)
        frame_number = 0
        current_time = time.time()
        while True:
            prev_frame_time = current_time
            # Capture frame
            frame = self.camera.capture_frame()

            # Check if the frame is valid
            if frame is None:
                frame_errors += 1
                print("Error capturing frame")
                time.sleep(1)
                if frame_errors > 5:
                    print("Too many frame errors, stopping...")
                    break
            elif frame_errors > 0:
                frame_errors = 0

            # Process frame
            animals = self.animal_recognizer.recognize_animal(frame)
            # Detect motion
            self.motion_detector.detect_motion(frame)

            current_time = time.time() # End of frame processing

            # Detect motion in the frame
            if self.motion_detector.get_motion_status():
                last_motion_time = current_time  # Update last motion time
                print("Motion detected, checking for animals...")

            if animals:
                new_animal = False
                # Draw bounding boxes around recognized animals
                for animal in animals:
                    if animal not in self.animals_seen:
                        new_animal = True
                    self.animals_seen.add(animal[0])  # Add animal class name to seen set
                    frame = self.animal_recognizer.draw_bounding_box(frame, animal)

                if not self.recording:
                    print("Animals detected, starting recording...")
                    self.recording = True
                    start_time = current_time
                    prev_frame_time = start_time

                    # Define video codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    containing_folder = f"{self.video_folder}/" if self.video_folder else ""
                    video_filename = f"{containing_folder}animal_recording_{timestamp}.mp4"
                    self.video_writer = cv2.VideoWriter(
                        video_filename, 
                        fourcc,
                        self.target_framerate,
                        (frame.shape[1], frame.shape[0])
                    )

                    # Insert video entry into the database
                    video_id = video_database.insert_video(video_filename, start_time, self.animals_seen)
                elif new_animal:
                    # Update the database with new animals
                    video_database.update_video_animals(video_id, self.animals_seen)

            # Write the frame to the video file if recording
            if self.recording:
                # Record frame
                num_frames = max(round(self.target_framerate * (current_time - prev_frame_time)), 1)
                for i in range(num_frames):
                    self.video_writer.write(frame)

                frame_number += 1
                frame_mod = (frame_number % 4) + 1
                print("." * frame_mod + " " * (5 - frame_mod), end="\r", flush=True)  # Print a dot for each frame recorded

                # Check for stop conditions
                elapsed_time = current_time - start_time
                video_database.update_video_duration(video_id, elapsed_time)

                # Explicit stop condition
                no_motion_condition = len(animals) == 0 and (int(current_time - last_motion_time) >= self.motion_timeout)
                max_time_condition = elapsed_time >= self.recording_duration

                if max_time_condition or no_motion_condition:
                    if no_motion_condition:
                        print(f"No motion detected for a while ({int(current_time - last_motion_time)} seconds), stopping recording...")
                    elif max_time_condition:
                        print("Max recording duration reached, stopping recording...")

                    print("\nRecording duration or motion timeout reached, stopping recording...")
                    print(f"{frame_number} frames recorded in {elapsed_time:.2f} seconds.")

                    # Update class state
                    self.recording = False
                    self.video_writer.release()
                    self.video_writer = None
                    self.animals_seen.clear()

                    # Update local state
                    motion_detected = False
                    start_time = None
                    video_filename = None
                    frame_number = 0

            else:
                # Reset motion_detected if no motion is detected for a while
                time.sleep(1)  # Check every 1 second
