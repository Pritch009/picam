import datetime
import os
import time
from motion_detection import MotionDetector
from animal_recognition import AnimalRecognizer
from threading import Event, Thread
from queue import Queue
import cv2
from datetime import datetime

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
        timeout=15,
        target_framerate=30.0,
        resolution=(1920, 1080)
    ):
        # Parameters
        self.resolution = resolution
        self.model_path = model_path
        self.threshold = threshold
        self.keywords = keywords
        self.recording_duration = recording_duration  # seconds
        self.timeout = timeout  # seconds without motion to stop recording
        self.video_folder = video_folder # Folder to save videos
        self.database_path =  database_path # Path to the SQLite database file
        self.target_framerate = target_framerate  # Target framerate for video recording
        # Components
        self.camera = HWCamera(resolution=resolution)
        # self.animal_recognizer = AnimalRecognizer(
        #     model_path=self.model_path, 
        #     keywords=self.keywords, 
        #     threshold=self.threshold
        # )
        self.queue = Queue()
        self.stop_condition_met = Event()
        self.start_condition_met = Event()
        # Configure later
        self.frames_to_recognize = 5  # Number of frames to utilize for initial recognition
        self.frames_between_recognition = 4  # Number of frames to skip between recognition
        self.frames_between_motion_detection = 1  # Number of frames to skip between motion detection
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
    
    def video_writer_and_process(self, start_time, queue, stop_event):
        video_writer = self.create_video_writer(start_time, self.resolution)
        frames_without_motion = 0
        motion_skip = 5
        frames_without_motion_limit = int(self.timeout * self.target_framerate / motion_skip)
        recording_frame_limit = self.recording_duration * self.target_framerate
        motion_detector = MotionDetector()
        frame_num = 0
        processing_time_queue = Queue()
        last_frame_time = start_time

        while True:
            if queue.empty():
                time.sleep(0.1)
                continue

            frame, frame_time = queue.get()
            frame_num += 1

            if frame_num % motion_skip == 0:
                # Process the frame before writing it
                process_start_time = time.time()
                motion_detected = motion_detector.detect_motion(frame)
                process_end_time = time.time()
                processing_time_queue.put(process_end_time - process_start_time)

                if motion_detected:
                    frames_without_motion = 0
                else:
                    frames_without_motion += 1
                    if frames_without_motion > frames_without_motion_limit:
                        print("No motion detected for a while, stopping recording...")
                        break

            num_frames = max(1, int(round((frame_time - last_frame_time) * self.target_framerate)))
            last_frame_time = frame_time

            # Write the frame to the video file
            for _ in range(num_frames):
                video_writer.write(frame)

            print(f"{frame_num}:{num_frames}" + "*" * (frame_num % 10) + " " * (10 - (frame_num % 10)), end="\r")
            
            # Check for stop conditions
            if frame_num >= recording_frame_limit:
                print("Max recording duration reached, stopping recording...")
                break

        stop_event.set()
        video_writer.release()
        print(f"Video recording stopped. {frame_num} frames recorded.")
        print(f"Average processing time: {sum(processing_time_queue.queue) / len(processing_time_queue.queue):.2f} seconds per frame processed.")

    
    def run_and_process(self):
        self.start_feed()
        print(f"Starting camera feed ({self.resolution[0]}x{self.resolution[1]})...")
        processing_time_queue = Queue()
        motion_detector = MotionDetector()
        stop_condition = Event()
        frame_time = None

        while True:
            while True:
                # Capture frame
                frame, frame_time = self.capture_frame("lores")
                processing_time_start = time.time()
                motion_detected = motion_detector.detect_motion(frame)
                if processing_time_queue.qsize() > 10:
                    processing_time_queue.get()
                
                processing_end_time = time.time()
                processing_time_queue.put(processing_end_time - processing_time_start)
                if motion_detected:
                    break
                else:
                    time.sleep(0.1)
                    continue

            queue = Queue(maxsize=100)
            # Start the video writer and processing in a separate thread
            Thread(
                target=self.video_writer_and_process,
                args=(frame_time, queue, stop_condition)
            ).start()

            time_to_capture = 1.0 / self.target_framerate

            # Start the frame capture loop
            while True:
                capture_start = time.time()
                capture = self.capture_frame("main")
                queue.put(capture)

                if stop_condition.is_set():
                    # Stop the frame capture loop
                    stop_condition.clear()
                    break

                capture_elapsed = time.time() - capture_start
                if capture_elapsed < time_to_capture:
                    time.sleep(time_to_capture - capture_elapsed)
            
    

        
    def run_motion_detection(self):
        self.start_feed()
        print("Starting motion detection...")
        lores_motion_detector = MotionDetector()
        while True:
            frame, _ = self.capture_frame("lores")
            if frame is None:
                print("Error capturing frame for motion detection")
                time.sleep(1)
                continue

            # Detect motion
            if lores_motion_detector.detect_motion(frame):
                print("Motion detected...")
                # Trigger the event
                Thread(target=self.process_frames).start()
                self.record_frames()
                lores_motion_detector.reset()

            time.sleep(0.1)  # Adjust the sleep time as needed

    def record_frames(self):
        print("starting recognition...")
        time_per_frame = 1.0 / self.target_framerate
        start_time = time.time()

        while True:
            # Capture frame
            frame, frame_time = self.capture_frame("main")
            
            # Put the frame in the queue
            self.queue.put((frame, frame_time))

            if self.stop_condition_met.is_set():
                # Stop recording frames
                self.stop_condition_met.clear()
                break

            if time_per_frame - (frame_time - start_time) > 0:
                time.sleep(time_per_frame - (frame_time - start_time))
            
            start_time = frame_time

    def process_frames(self):
        start_time  = time.time()
        stop = False
        frame_count = 0
        animals = []
        video_writer = None
        last_motion_time = start_time
        last_recognition_time = start_time
        recognition_times = []
        motion_detection_times = []

        while not stop:
            # Check if there are frames in the queue
            if self.queue.empty():
                if time.time() - start_time > 2:
                    # If no frames for 2 seconds, stop processing
                    stop = True
                time.sleep(0.1)
                continue

            frame, frame_time = self.queue.get()
            frame_count += 1

            # Run motion detection
            if frame_count % self.frames_between_motion_detection == 0:
                motion_detection_time_start = time.time()
                motion_detected = self.motion_detector.detect_motion(frame)
                motion_detection_times.append(time.time() - motion_detection_time_start)
                print(f"Motion detection took {time.time() - motion_detection_time_start:.2f} seconds.")
                if motion_detected:
                    last_motion_time = time.time()
            
            # Motion was detected already, lets check for animals every X frames
            if frame_count % self.frames_between_recognition == 0:
                recognition_start = time.time()
                animals = self.animal_recognizer.recognize_animal(frame)
                recognition_times.append(time.time() - recognition_start)
                print(f"Recognized {len(animals)} animals in {time.time() - recognition_start:.2f} seconds.")

                if len(animals) > 0:
                    last_recognition_time = time.time()
                    if video_writer is None:
                        video_writer = self.create_video_writer(frame_time, frame.shape)
                        start_time = frame_time
                
            # if recording 
            if video_writer is not None:
                # Draw bounding boxes around recognized animals
                frame = self.animal_recognizer.draw_bounding_boxes(frame, animals)

                # Write the frame to the video file
                video_writer.write(frame)

                # Check for stop conditions
                elapsed_time_condition = time.time() - start_time >= self.recording_duration
                recog_condition = frame_time - last_recognition_time >= self.timeout
                motion_condition = frame_time - last_motion_time >= self.timeout
                if elapsed_time_condition or motion_condition or recog_condition:
                    # Stop recording
                    video_writer.release()
                    stop = True

                    if motion_condition:
                        print(f"No motion detected for a while ({int(frame_time - last_motion_time)} seconds), stopping recording...")
                    elif elapsed_time_condition:
                        print("Max recording duration reached, stopping recording...")
                    elif recog_condition:
                        print("No animals detected for a while, stopping recording...")

                    print(f"{frame_count} frames recorded in {(frame_time - start_time):.2f} seconds.")     
                    print(f"Average recognition time: {sum(recognition_times) / len(recognition_times):.2f} seconds per frame processed.")        
                    print(f"Average motion detection time: {sum(motion_detection_times) / len(motion_detection_times):.2f} seconds per frame processed.")
            elif video_writer is None and (frame_time - last_motion_time) > 2:
                # If no animals detected for 2 seconds, stop processing
                stop = True
                print("No animals detected for 2 seconds, stopping processing...")

        self.stop_condition_met.set()

    def create_video_writer(self, start_time, resolution):
        # Create a timestamp for the video filename
        time_str = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.video_folder}/animal_recording_{time_str}_{resolution[0]}x{resolution[1]}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            filename, 
            fourcc,
            self.target_framerate,
            (self.resolution[0], self.resolution[1]),
        )
        return video_writer