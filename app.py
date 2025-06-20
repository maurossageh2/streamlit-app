import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import re

# Set wide layout
st.set_page_config(layout="wide")

# Title
st.title("Resistor Detection and Classification")

# Introduction
st.markdown("""
Detect resistors and their values in an uploaded image.

### Tips for Better Results
To ensure accurate detection and classification:
- **Good Lighting**: Use bright, even lighting to avoid shadows that can obscure color bands.
- **High-Quality Images**: Upload clear, high-resolution images for better color accuracy.
- **Proper Distance**: Maintain a reasonable **distance between resistors** to avoid overlapping detections.
- **Clear Background**: Use a plain background to minimize distractions and improve detection accuracy.
""")

# Function to parse resistance values with k and M suffixes
def parse_resistance(value):
    if value == "Unknown":
        return None
    try:
        match = re.match(r'^(\d*\.?\d+)([kM]?)$', value.strip(), re.IGNORECASE)
        if not match:
            return None
        num, suffix = match.groups()
        num = float(num)
        if suffix.lower() == 'k':
            return num * 1e3
        elif suffix.lower() == 'm':
            return num * 1e6
        else:
            return num
    except (ValueError, AttributeError):
        return None

# Function to format resistance values for display
def format_resistance(value):
    if value is None:
        return "Invalid"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M ohms"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}k ohms"
    else:
        return f"{value:.2f} ohms"

# Load models
@st.cache_resource
def load_models():
    model_resistor = YOLO("./models/resistor_yolov8_v2.pt")
    model_value = YOLO("./models/resistor_bands_yolov8.pt")
    return model_resistor, model_value

model_resistor, model_value = load_models()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}

# Upload multiple images
uploaded_files = st.file_uploader("Upload images containing resistors", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Clear previous results and process new files when uploaded
if uploaded_files:
    # Reset session state for results
    st.session_state.results = {}
    
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        with st.spinner(f"Detecting resistors in {file_key}..."):
            # Read and convert image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect resistors
            results = model_resistor.predict(image_rgb)
            boxes = results[0].boxes

            rois = []
            coords = []
            resistor_values = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls = int(box.cls.item())

                if cls == 0 and conf > 0.15:
                    roi = image_rgb[y1:y2, x1:x2]
                    rois.append(roi)
                    coords.append((x1, y1, x2, y2))

            for i, roi in enumerate(rois):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(temp_file.name, roi_bgr)
                    result = model_value(temp_file.name)
                    temp_file_path = temp_file.name

                pred_box = result[0].boxes
                if len(pred_box) > 0:
                    top_pred = pred_box[0]
                    cls_id = int(top_pred.cls.item())
                    label = result[0].names[cls_id]
                    resistor_values.append(label)
                else:
                    resistor_values.append("Unknown")

                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

            # Annotate image with IDs and Values
            annotated_image = image_rgb.copy()
            for idx, (x1, y1, x2, y2), label in zip(range(1, len(coords) + 1), coords, resistor_values):
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 0), 5)
                
                # Convert OpenCV image to PIL for Unicode text rendering
                pil_image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(pil_image)
                
                # Try to use Arial, fallback to DejaVu Sans or default sans-serif
                try:
                    font = ImageFont.truetype("arial.ttf", 70)
                except IOError:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 70)
                    except IOError:
                        font = ImageFont.load_default()
                
                # Draw ID in black
                id_text = f"{idx}:"
                id_position = (x1, y1 - 80)
                draw.text(id_position, id_text, font=font, fill=(0, 0, 0))
                
                # Calculate position for value based on ID text width
                id_width = draw.textlength(id_text, font=font)
                value_position = (x1 + id_width, y1 - 80)
                
                # Draw value and ohm symbol in dark blue
                value_text = f"{label}Ω"
                draw.text(value_position, value_text, font=font, fill=(0, 0, 128))
                
                # Convert back to OpenCV format
                annotated_image = np.array(pil_image)

            # Store results in session state
            st.session_state.results[file_key] = {
                'image_rgb': image_rgb,
                'annotated_image': annotated_image,
                'resistor_values': resistor_values,
                'num_resistors': len(resistor_values)
            }

# Display results for each processed file
for file_key, result in st.session_state.results.items():
    st.subheader(f"Results for {file_key}")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(result['image_rgb'], caption="Original Image", use_container_width=True)
    with col_img2:
        st.image(result['annotated_image'], caption=f"Detected {result['num_resistors']} resistors", use_container_width=True)

    # Show table
    resistor_values = result['resistor_values']
    if resistor_values:
        st.subheader("Detected Resistors")
        data = {
            "ID": list(range(1, len(resistor_values) + 1)),
            "Value": resistor_values
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Resistance calculation form
        st.subheader("Resistance Calculation")
        with st.form(key=f"calc_form_{file_key}"):
            calculation_type = st.selectbox(
                "Select calculation type",
                ["Series", "Parallel"],
                key=f"calc_type_{file_key}"
            )
            valid_indices = [i for i, val in enumerate(resistor_values) if parse_resistance(val) is not None]
            available_ids = [str(i + 1) for i in valid_indices]
            selected_ids = st.multiselect(
                "Select resistor IDs for calculation",
                available_ids,
                key=f"select_ids_{file_key}"
            )
            calculate_button = st.form_submit_button("Calculate")

            if calculate_button and selected_ids:
                indices = [int(id) - 1 for id in selected_ids]
                selected_values = [parse_resistance(resistor_values[i]) for i in indices if parse_resistance(resistor_values[i]) is not None]

                if selected_values:
                    if calculation_type == "Series":
                        total_resistance = sum(selected_values)
                        st.write(f"Total resistance in **series**: **{format_resistance(total_resistance)}**")
                    else:  # Parallel
                        try:
                            inverse_sum = sum(1 / val for val in selected_values)
                            total_resistance = 1 / inverse_sum
                            st.write(f"Total resistance in **parallel**: **{format_resistance(total_resistance)}**")
                        except ZeroDivisionError:
                            st.error("Cannot calculate parallel resistance with zero-ohm resistors.")
                else:
                    st.warning("No valid resistor values selected for calculation.")
    else:
        st.write("No resistors detected.")