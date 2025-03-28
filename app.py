import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO
import tempfile
import torch
import cvzone
from paddleocr import PaddleOCR
from PIL import Image

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Set device (Change to 'cuda' for NVIDIA GPU or 'cpu' for CPU processing)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (Ensure they match your training labels)
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Initialize PaddleOCR for number plate recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Streamlit UI
st.title("Helmet & Number Plate Detection")
st.write("Upload an **image or video** to detect helmets, riders, and number plates.")

# Upload options
option = st.sidebar.radio("Choose Input Type:", ["Upload Image", "Upload Video"])

# Image Processing Function
def detect_objects(image):
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    model.to(device)
    results = model(new_img, stream=True)

    detected_image = image.copy()
    detected_number_plate = None

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls

        # Convert to tensor and move to device
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box[4] * 100)) / 100  # Confidence score
            cls = int(box[5])

            if 0 <= cls < len(classNames):
                label = classNames[cls]

                # Define colors for different detections
                color_map = {
                    "with helmet": (0, 255, 0),  # Green
                    "without helmet": (0, 0, 255),  # Red
                    "rider": (255, 165, 0),  # Orange
                    "number plate": (255, 255, 0)  # Cyan
                }

                # Draw bounding box
                cvzone.cornerRect(detected_image, (x1, y1, w, h), l=15, rt=5, colorR=color_map[label])
                cvzone.putTextRect(detected_image, f"{label.upper()} ({conf*100:.1f}%)", (x1, y1 - 10),
                                   scale=1.5, offset=10, thickness=2, colorT=(0, 0, 0), colorR=color_map[label])

                # OCR for Number Plate Recognition
                if label == "number plate":
                    crop = image[y1:y2, x1:x2]
                    try:
                        ocr_result = ocr.ocr(crop, cls=True)
                        if ocr_result and ocr_result[0]:
                            number_plate_text = ocr_result[0][0][1][0]  # Extract detected text
                            detected_number_plate = number_plate_text

                            # Display extracted text on image
                            cvzone.putTextRect(detected_image, f"Plate: {number_plate_text}",
                                               (x1, y1 - 50), scale=1.5, offset=10,
                                               thickness=2, colorT=(0, 0, 0), colorR=(0, 255, 255))
                            print(f"Detected Number Plate: {number_plate_text}")

                    except Exception as e:
                        print(f"Error in OCR: {e}")

    return detected_image, detected_number_plate


# **Image Upload & Processing**
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Detect objects
        detected_image, number_plate_text = detect_objects(image)

        # Display original and detected images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detected Image", use_column_width=True)

        # Display number plate text if detected
        if number_plate_text:
            st.markdown(f"### 🏷️ Detected Number Plate: **{number_plate_text}**")

# **Video Upload & Processing**
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())

        # Open video
        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Detect objects in frame
            detected_frame, number_plate_text = detect_objects(frame)

            # Convert frame for display
            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
