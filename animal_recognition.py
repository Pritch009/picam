import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


class AnimalRecognizer:
    def __init__(
        self,
        model_path="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", 
        local_model_dir="mega_detector_model",
        keywords=["cat", "man"],
        threshold=0.3
    ):
        self.model_path = model_path
        self.local_model_dir = local_model_dir
        self.keywords = keywords
        self.threshold = threshold
        self.model = None
        self.load_model()
        self.load_class_name_map()

    def load_model(self):
        print("Loading model...")
        self.model = hub.load(self.model_path)
        print("Model downloaded.")

    def load_class_name_map(self, class_names_path="./oidv6-class-descriptions.csv"):
        # Load the class name map from a csv file
        class_names = {}
        with open(class_names_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    class_id = parts[0]
                    class_name = parts[1]
                    class_names[class_id] = class_name
        self.class_names = class_names
        print("Class name map loaded.")


    def recognize_animal(self, frame):
        if self.model is None:
            print("Model not loaded.")
            return []

        # Convert the frame to RGB (if it's not already)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to a fixed size (e.g., 640x480)
        resized_frame = cv2.resize(rgb_frame, (640, 480))

        # Normalize pixel values to be in the range [0.0, 1.0]
        normalized_frame = resized_frame / 255.0
        
        # Expand dimensions since the model expects images to have shape: [1, height, width, 3]
        input_tensor = np.expand_dims(normalized_frame, 0).astype(np.float32)

        # Perform the object detection
        detections = self.model.signatures['default'](tf.constant(input_tensor))

        # Extract detection boxes, scores, class names, and class labels
        boxes = detections['detection_boxes'].numpy()
        scores = detections['detection_scores'].numpy()
        class_names = detections['detection_class_names'].numpy()
        class_labels = detections['detection_class_labels'].numpy()

        num_detections = scores.shape[0]

        # Filter detections based on a confidence threshold (e.g., 30%)
        animal_detections = []
        for i in range(num_detections):
            if scores[i] > self.threshold:
                class_name_raw = class_names[i].decode('utf-8')  # Decode bytes to string
                class_name = self.class_names.get(class_name_raw, "").lower()
                if class_name == "":
                    continue

                # Check if the class name contains keywords for detection
                if any(keyword in class_name.lower() for keyword in self.keywords):
                    box = boxes[i]
                    ymin, xmin, ymax, xmax = box
                    im_height, im_width, _ = frame.shape  # Use original frame dimensions
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                    ymin * im_height, ymax * im_height)
                    animal_detections.append((class_name, int(left), int(top), int(right-left), int(bottom-top)))  # x, y, width, height

        return animal_detections

    def draw_bounding_box(self, frame, boxes):
        # Draw bounding boxes around recognized animals on the frame
        # Check if boxes is a tuple or a list of tuples
        if isinstance(boxes, tuple):
            boxes = [boxes]
        elif not isinstance(boxes, list):
            raise ValueError("boxes should be a tuple or a list of tuples")
        
        # Draw rectangles around the detected animals
        for box in boxes:
            class_name, x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw name under the rectangle
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame