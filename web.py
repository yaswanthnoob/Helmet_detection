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
import os

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Streamlit UI Configuration
st.set_page_config(page_title="Helmet & Number Plate Detection", page_icon="🛵", layout="wide")

st.sidebar.title("⚙️ Options")
option = st.sidebar.radio("📂 Choose Input Type:", ["Upload Image", "Upload Video"])

# Detection Function
def detect_objects(image):
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model.to(device)
    results = model(new_img, stream=True)

    detected_image = image.copy()
    detected_number_plate = None
    detect_plate = False
    helmet_detected = False
    no_helmet_detected = False

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

        number_plate_coords = None

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])
            
            if 0 <= cls < len(classNames):
                label = classNames[cls]
                
                color_map = {
                    "with helmet": (0, 255, 0),
                    "without helmet": (0, 0, 255),
                    "rider": (255, 165, 0),
                    "number plate": (255, 255, 0)
                }
                
                cvzone.cornerRect(detected_image, (x1, y1, w, h), l=15, rt=5, colorR=color_map[label])
                cvzone.putTextRect(detected_image, f"{label.upper()} ({conf*100:.1f}%)", (x1, y1 - 10),
                                   scale=1.5, offset=10, thickness=2, colorT=(0, 0, 0), colorR=color_map[label])
                
                if label == "without helmet":
                    detect_plate = True
                    no_helmet_detected = True
                if label == "with helmet":
                    helmet_detected = True
                
                if label == "number plate":
                    number_plate_coords = (x1, y1, x2, y2)

        if detect_plate and number_plate_coords:
            x1, y1, x2, y2 = number_plate_coords
            crop = image[y1:y2, x1:x2]

            # OCR Processing
            try:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ocr_result = ocr.ocr(crop_gray, cls=True)
                if ocr_result and ocr_result[0]:
                    detected_number_plate = ocr_result[0][0][1][0]
                    cvzone.putTextRect(detected_image, f"Plate: {detected_number_plate}",
                                       (x1, y1 - 50), scale=1.5, offset=10,
                                       thickness=2, colorT=(0, 0, 0), colorR=(0, 255, 255))
            except Exception as e:
                print(f"Error in OCR: {e}")

    return detected_image, detected_number_plate, detect_plate, helmet_detected, no_helmet_detected

# Image Upload & Processing
if option == "Upload Image":
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        detected_image, number_plate_text, detect_plate, helmet_detected, no_helmet_detected = detect_objects(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🖼️ Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="🎯 Detected Image", use_column_width=True)

        # Detection Summary
        st.markdown("### 📝 Detection Summary:")
        st.write(f"- 🪖 Helmet Detected: {'✅ Yes' if helmet_detected else '❌ No'}")
        st.write(f"- 🚫 Without Helmet Detected: {'✅ Yes' if no_helmet_detected else '❌ No'}")
        st.write(f"- 🏷️ Number Plate: **{'Detected: ' + number_plate_text if number_plate_text else 'Not Detected'}**")

        # Download Processed Image
        output_filename = "processed_image.jpg"
        cv2.imwrite(output_filename, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
        with open(output_filename, "rb") as file:
            st.download_button("📥 Download Processed Image", file, file_name="processed_image.jpg")

# Video Upload & Processing
elif option == "Upload Video":
    uploaded_video = st.file_uploader("📹 Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        # Save output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = "processed_video.mp4"
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            detected_frame, number_plate_text, detect_plate, helmet_detected, no_helmet_detected = detect_objects(frame)
            out.write(detected_frame)

            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        out.release()

        # Download Processed Video
        with open(output_filename, "rb") as file:
            st.download_button("📥 Download Processed Video", file, file_name="processed_video.mp4")
