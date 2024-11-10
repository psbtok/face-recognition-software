import cv2
import numpy as np
import json
import time
import os
import dlib
from sort import SortTracker  # Import SortTracker from sort-tracker

# Paths to the YOLO configuration, weights, and coco names files
config_path = 'yolov3.cfg'  # Update with your path to yolov3.cfg
weights_path = 'yolov3.weights'  # Update with your path to yolov3.weights
class_names_path = 'coco.names'  # Update with your path to coco.names

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load class names
with open(class_names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Load facial recognition data
with open('facial_features.json', 'r') as f:
    facial_data = json.load(f)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")  # Ensure this file is present

# Initialize SORT tracker
tracker = SortTracker(max_age=5, min_hits=3, iou_threshold=0.2)

# Define thresholds
confidence_threshold = 0.5
recognition_threshold = 5000  # Threshold for facial recognition similarity

# Option to flip the image horizontally
flip_horizontally = True

# Function to calculate the Euclidean distance between two sets of landmarks
def calculate_distance(landmarks1, landmarks2):
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)
    return np.linalg.norm(landmarks1 - landmarks2)

# Function to extract facial landmarks from an image
def extract_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    landmarks_list = []
    
    for face in faces:
        if face.width() <= 0 or face.height() <= 0:
            continue  # Skip invalid face regions
        
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        landmarks_list.append(landmarks_points)
        
    return landmarks_list

# Function to recognize the person based on facial features
def recognize_person(landmarks):
    min_distance = 9999
    recognized_name = None
    
    # Compare the detected landmarks with the stored data
    for name, data in facial_data.items():
        stored_landmarks = data['landmarks']
        for stored_landmark in stored_landmarks:
            distance = calculate_distance(landmarks, stored_landmark)
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
        print(name, distance)
    
    if min_distance < recognition_threshold:
        return recognized_name
    return None

# Function to detect people using YOLO
def detect_people(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_id == 0:  # Class ID 0 is 'person'
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    result_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            result_boxes.append(boxes[i])

    return result_boxes

# Dictionary to keep track of currently visible IDs and recognized names
recognized_names = {}

# Function to track and draw bounding boxes with IDs
# Function to track and draw bounding boxes with IDs
def track_and_draw(image, boxes):
    if len(boxes) == 0:
        return  # If no boxes are detected, skip updating the tracker

    dets = np.array([[x, y, x + w, y + h, 1.0, 0] for (x, y, w, h) in boxes])  # Added class ID as 0
    tracked_objects = tracker.update(dets, 2)

    global recognized_names

    for obj in tracked_objects:
        if len(obj) >= 5:
            x1, y1, x2, y2, obj_id = map(int, obj[:5])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the person has been recognized before
            if obj_id not in recognized_names:
                # Extract face and recognize person
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    landmarks_list = extract_facial_features(face)
                    for landmarks in landmarks_list:
                        recognized_name = recognize_person(landmarks)
                        if recognized_name:
                            recognized_names[obj_id] = recognized_name
                            cv2.putText(image, f'Name: {recognized_name}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If the person was recognized before, display the name
            if obj_id in recognized_names:
                cv2.putText(image, f'Name: {recognized_names[obj_id]}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# Try different camera indices to find an available webcam
cap = None
for i in range(5):  # Try camera indices 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any webcam.")
    exit()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Flip frame horizontally if enabled
    if flip_horizontally:
        frame = cv2.flip(frame, 1)

    # Detect people (class_id == 0) in the current frame
    boxes = detect_people(frame)
    
    # Track and recognize people
    track_and_draw(frame, boxes)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
