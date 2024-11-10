import cv2
import numpy as np
import os
import time
import json
import dlib

# Option to flip the image horizontally
flip_horizontally = True  # Set to True to enable horizontal flip

# Paths to the YOLO configuration, weights, and coco names files
config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
class_names_path = 'coco.names'

# Check if the class names file exists
if not os.path.exists(class_names_path):
    raise FileNotFoundError(f"{class_names_path} file not found. Please make sure it exists in the specified directory.")

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

# Define the confidence threshold for YOLO and facial recognition
confidence_threshold = 0.5
recognition_threshold = 5000  # Threshold for facial recognition similarity

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
        # Check if the face region is valid
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
    # If the distance is below the threshold, recognize the person
    if min_distance < recognition_threshold:
        return recognized_name
    return None

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
    result_class_ids = []
    if len(indices) > 0:
        for i in indices.flatten():
            result_boxes.append(boxes[i])
            result_class_ids.append(class_ids[i])

    return result_boxes, result_class_ids

def draw_boxes(image, boxes, class_ids, last_recognized_names):
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = str(classes[class_id])  # Get the class name (which will be 'person')
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw the recognized name for each person
        if last_recognized_names.get((x, y, w, h)):
            cv2.putText(image, f"Recognized: {last_recognized_names[(x, y, w, h)]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

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

# Variables for real-time frame rate calculation
prev_time = 0

# Define the interval in seconds for capturing a frame
capture_interval = 1  # Capture a frame every 1 second (change this value for different intervals)

# Initialize frame count
frame_count = 0  # Initialize frame count

# Variable to store the last recognized names for each person
last_recognized_names = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    if frame_count % capture_interval == 0:  # Capture frame at the specified interval
        # Flip the frame horizontally if the option is enabled
        if flip_horizontally:
            frame = cv2.flip(frame, 1)
        
        # Detect people (class_id == 0) in the current frame
        boxes, class_ids = detect_people(frame)

        # Extract facial features for each detected person
        for (x, y, w, h) in boxes:
            face = frame[y:y + h, x:x + w]
            if face.size == 0:  # Skip if face region is invalid
                continue
            
            landmarks_list = extract_facial_features(face)
            for landmarks in landmarks_list:
                # Recognize person based on facial landmarks
                print("recognizing person")
                recognized_name = recognize_person(landmarks)
                if recognized_name:
                    last_recognized_names[(x, y, w, h)] = recognized_name  # Store name by bounding box
        
        # Display recognized images after processing
        if last_recognized_names:  # If any name has been recognized
            output_frame = draw_boxes(frame, boxes, class_ids, last_recognized_names)
            cv2.imshow('Recognized People', output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
