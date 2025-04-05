import datetime
import os
import time
from motion_detection import MotionDetector
from animal_recognition import AnimalRecognizer
from threading import Event, Thread
import cv2

from video_database import VideoDatabase

use_mock_camera = os.environ.get('USE_MOCK_CAMERA', 'False').lower() == 'true'

if use_mock_camera:
    from mock_camera import MockCamera as HWCamera
    print("Using MockCamera")
else:
    try:
        from pi_camera import Camera as HWCamera
        print("Using Raspberry Pi camera")
    except ImportError:
        print("picamera not found.  Using MockCamera.  Set environment variable USE_MOCK_CAMERA=TRUE to suppress this message.")
        from mock_camera import MockCamera as HWCamera

class RichCamera:
    def __init__(
        self,
        model_path="./model/mobilenetv2_ssd_fixed_1920_1080.tflite",
        video_folder="videos",
        database_path="video_database.db",
        keywords=["man"],
        threshold=0.3,
        recording_duration=300,
        motion_timeout=15,
        target_framerate=30.0,
        resolution=(1920, 1080)
    ):
        # Components
        self.resolution = resolution
        self.camera = HWCamera(resolution=resolution)
        self.animal_recognizer = AnimalRecognizer(
            model_path=self.model_path, 
            keywords=self.keywords, 
            threshold=self.threshold
        )
        self.motion_detector = MotionDetector()
        # Parameters
        self.recording_duration = recording_duration  # seconds
        self.motion_timeout = motion_timeout  # seconds without motion to stop recording
        self.video_folder = video_folder # Folder to save videos
        self.database_path =  database_path # Path to the SQLite database file
        self.target_framerate = target_framerate  # Target framerate for video recording
        # Configure later
        self.frames_to_recognize = 5  # Number of frames to utilize for initial recognition
        self.frames_between_recognition = 4  # Number of frames to skip between recognition
        # State
        self.recording = False
        self.video_writer = None
        self.last_motion_time = None  # Track the last time motion was detected
        self.animals_seen = set()  # Track unique animals seen

    def start_feed(self):
        self.camera.start_feed()
        print("Camera feed started")

    def stop_feed(self):
        self.camera.stop_feed()
        print("Camera feed stopped")
    
    def close(self):
        self.camera.close()
        print("Camera closed")

    def capture_frame(self, camera="main"):
        frame = self.camera.capture_frame(camera=camera)
        frame_recorded_time = time.time()

        if frame is None:
            raise Exception("Error capturing frame")
        
        return (frame, frame_recorded_time)
        

    
    def run_motion_detection(self):
        self.start_feed()
        print("Starting motion detection...")
        while True:
            frame = self.capture_frame("lores")
            if frame is None:
                print("Error capturing frame for motion detection")
                time.sleep(1)
                continue
            # Detect motion
            if self.motion_detector.detect_motion(frame):
                print("Motion detected...")
                # Trigger the event
                self.run_recognition()

            time.sleep(0.1)  # Adjust the sleep time as needed

    def create_video_writer(self, start_time):
        # Create a timestamp for the video filename
        filename = f"{self.video_folder}/animal_recording_{start_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            filename, 
            fourcc,
            self.target_framerate,
            (self.resolution[0], self.resolution[1]),
        )
        return video_writer

    def run_recognition(self):
        print("starting recognition...")

        # if an animal is detected, start recording
        recording = False
        start_time = time.time()
        last_motion_time = time.time()

        # first, check if any animals can be detected for a handful of frames
        frame_count = self.frames_to_recognize
        frame = None
        frame_recorded_time = None
        animals = []
        video_writer = None
        while frame_count < self.frames_to_recognize:
            frame, frame_recorded_time = self.capture_frame("main")
            frame_count += 1

            if frame is None:
                raise Exception("Error capturing frame for recognition")
            
            # Detect animals
            animals = self.animal_recognizer.recognize_animal(frame)
            if len(animals) > 0:
                print("Animals detected!")
                recording = True
                break

        while recording:
            video_writer = self.create_video_writer(start_time)
            self.animal_recognizer.draw_bounding_box(frame, animals)
            if self.motion_detector.detect_motion(frame):
                last_motion_time = frame_recorded_time

            # Check if the recording duration has been reached
            elapsed_time = frame_recorded_time - start_time
            if elapsed_time >= self.recording_duration:
                print("Max recording duration reached, stopping recording...")
                recording = False
                break

            # Check if no motion has been detected for a while
            if frame_recorded_time - last_motion_time >= self.motion_timeout:
                print("No motion detected for a while, stopping recording...")
                recording = False
                break

            # Write the frame to the video file    
            video_writer.write(frame)

            # Capture next frame
            frame = self.capture_frame("main")
        
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved successfully to {video_writer.get_filename()}")





    def run_in_background(self):
        print("Starting camera...")
        start_time = None
        last_motion_time = time.time()
        video_id = None
        frame_errors = 0
        video_database = VideoDatabase(db_name=self.database_path)
        frame_number = 0
        current_time = time.time()
        self.camera.start_feed()
        while True:
            prev_frame_time = current_time
            # Capture frame
            frame = self.camera.capture_frame("main" if self.recording else "lores")

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
                print("Motion detected...")

            if animals:
                new_animal = False
                # Draw bounding boxes around recognized animals
                for animal in animals:
                    if animal not in self.animals_seen:
                        new_animal = True
                    self.animals_seen.add(animal[0])  # Add animal class name to seen set
                    frame = self.animal_recognizer.draw_bounding_box(frame, animal)

                if not self.recording:
                    print("Animals detected, now recording...")
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
                        (self.resolution[0], self.resolution[1]),
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

                    print(f"{frame_number} frames recorded in {elapsed_time:.2f} seconds.")

                    # Update class state
                    self.recording = False
                    self.video_writer.release()
                    self.video_writer = None
                    self.animals_seen.clear()

                    # Update local state
                    start_time = None
                    video_filename = None
                    frame_number = 0
                    last_motion_time = current_time

            else:
                # Reset motion_detected if no motion is detected for a while
                time.sleep(1)  # Check every 1 second
