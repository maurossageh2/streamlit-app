import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import requests

# Title
st.title("Resistor Detection and Classification")

# Load models
@st.cache_resource
def load_models():
    model_resistor = YOLO("./models/resistor_yolov8.pt")
    model_value = YOLO("./models/resistor_bands_yolov8.pt")
    return model_resistor, model_value

model_resistor, model_value = load_models()

# Upload image
uploaded_file = st.file_uploader("Upload an image containing resistors", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    # Detect resistors
    with st.spinner("Detecting resistors..."):
        results = model_resistor.predict(image_rgb)
        boxes = results[0].boxes

        rois = []
        coords = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())

            if cls == 0 and conf > 0.15:
                roi = image_rgb[y1:y2, x1:x2]
                rois.append(roi)
                coords.append((x1, y1, x2, y2))

        resistor_values = []

        for i, roi in enumerate(rois):
            # Save ROI to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_file.name, roi_bgr)
                result = model_value(temp_file.name)

            pred_box = result[0].boxes
            if len(pred_box) > 0:
                top_pred = pred_box[0]
                cls_id = int(top_pred.cls.item())
                label = result[0].names[cls_id]
                resistor_values.append(label)
            else:
                resistor_values.append("Unknown")

        # Annotate image
        annotated_image = image_rgb.copy()
        for (x1, y1, x2, y2), label in zip(coords, resistor_values):
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3)

        # Show results
        st.image(annotated_image, caption=f"Detected {len(resistor_values)} resistors", use_container_width=True)
