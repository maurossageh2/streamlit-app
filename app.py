import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import re

# Configurar diseño ancho
st.set_page_config(layout="wide")

# Título
st.title("Detección y Clasificación de Resistencias")

# Introducción
st.markdown("""
Detecta resistencias y sus valores en una imagen cargada.

### Consejos para Mejores Resultados
Para garantizar una detección y clasificación precisas:
- **Buena Iluminación**: Usa una iluminación brillante (**Fotos con flash**) y uniforme para evitar sombras que oculten las bandas de color.
- **Imágenes de Alta Calidad**: Carga imágenes claras y de alta resolución para una mejor precisión en los colores.
- **Distancia Adecuada**: Mantén una **distancia razonable entre resistencias** para evitar superposiciones en la detección.
- **Fondo Limpio**: Usa un fondo simple para minimizar distracciones y mejorar la precisión de la detección.
""")

# Función para parsear valores de resistencia con sufijos k y M
def parse_resistance(value):
    if value == "Desconocido":
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

# Función para formatear valores de resistencia para mostrar
def format_resistance(value):
    if value is None:
        return "Inválido"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M ohmios"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}k ohmios"
    else:
        return f"{value:.2f} ohmios"

# Cargar modelos
@st.cache_resource
def load_models():
    model_resistor = YOLO("./models/resistor_yolov8_v2.pt")
    model_value = YOLO("./models/resistor_bands_yolov8.pt")
    return model_resistor, model_value

model_resistor, model_value = load_models()

# Inicializar estado de la sesión
if 'results' not in st.session_state:
    st.session_state.results = {}

# Cargar múltiples imágenes
uploaded_files = st.file_uploader("Carga imágenes que contengan resistencias", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Limpiar resultados Belinda
if uploaded_files:
    # Resetear estado de la sesión para resultados
    st.session_state.results = {}
    
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        with st.spinner(f"Detectando resistencias en {file_key}..."):
            # Leer y convertir imagen
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detectar resistencias
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
                    # Ajustes para reducir "Desconocido"
                    result = model_value(temp_file.name, conf=0.1, max_det=5)
                    temp_file_path = temp_file.name

                pred_box = result[0].boxes
                label = "Desconocido"
                conf_score = None

                if len(pred_box) > 0:
                    # Obtener la mejor predicción
                    top_pred = sorted(pred_box, key=lambda x: x.conf.item(), reverse=True)[0]
                    cls_id = int(top_pred.cls.item())
                    label = result[0].names[cls_id]
                    conf_score = top_pred.conf.item()

                resistor_values.append({
                    'label': label,
                    'conf_score': conf_score
                })

                # Limpiar archivo temporal
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

            # Anotar imagen con IDs y valores
            annotated_image = image_rgb.copy()
            image_height, image_width = image_rgb.shape[:2]

            # Definir tamaño de fuente y grosor relativo a las dimensiones de la imagen
            base_font_size = max(20, min(100, int(min(image_width, image_height) * 0.03)))
            base_thickness = max(2, min(10, int(min(image_width, image_height) * 0.004)))

            for idx, (x1, y1, x2, y2), res_val in zip(range(1, len(coords) + 1), coords, resistor_values):
                # Dibujar cuadro delimitador
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 0), base_thickness)
                
                # Convertir imagen OpenCV a PIL para renderizar texto
                pil_image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(pil_image)
                
                # Cargar fuente
                try:
                    font = ImageFont.truetype("arial.ttf", base_font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", base_font_size)
                    except IOError:
                        font = ImageFont.load_default()
                        if base_font_size > 20:
                            font.size = base_font_size
                
                # Dibujar ID en negro
                id_text = f"{idx}:"
                id_position = (x1, y1 - int(base_font_size * 1.2))
                draw.text(id_position, id_text, font=font, fill=(0, 0, 0))
                
                # Calcular posición para el valor
                id_width = draw.textlength(id_text, font=font)
                value_position = (x1 + id_width, y1 - int(base_font_size * 1.2))
                
                # Dibujar valor y símbolo de ohmios en azul oscuro
                value_text = f"{res_val['label']}Ω"
                draw.text(value_position, value_text, font=font, fill=(0, 0, 128))
                
                # Convertir de vuelta a formato OpenCV
                annotated_image = np.array(pil_image)

            # Guardar resultados en el estado de la sesión
            st.session_state.results[file_key] = {
                'image_rgb': image_rgb,
                'annotated_image': annotated_image,
                'resistor_values': resistor_values,
                'num_resistors': len(resistor_values)
            }

# Mostrar resultados para cada archivo procesado
for file_key, result in st.session_state.results.items():
    st.subheader(f"Resultados para {file_key}")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(result['image_rgb'], caption="Imagen Original", use_container_width=True)
    with col_img2:
        st.image(result['annotated_image'], caption=f"{result['num_resistors']} resistencias detectadas", use_container_width=True)

    # Mostrar tabla
    resistor_values = result['resistor_values']
    if resistor_values:
        st.subheader("Resistencias Detectadas")
        data = {
            "ID": list(range(1, len(resistor_values) + 1)),
            "Valor": [rv['label'] for rv in resistor_values],
            "Confianza": [f"{rv['conf_score']:.3f}" if rv['conf_score'] is not None else "N/A" for rv in resistor_values]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Formulario de cálculo de resistencia
        st.subheader("Cálculo de Resistencia")
        with st.form(key=f"calc_form_{file_key}"):
            calculation_type = st.selectbox(
                "Selecciona el tipo de cálculo",
                ["Serie", "Paralelo"],
                key=f"calc_type_{file_key}"
            )
            valid_indices = [i for i, val in enumerate([rv['label'] for rv in resistor_values]) if parse_resistance(val) is not None]
            available_ids = [str(i + 1) for i in valid_indices]
            selected_ids = st.multiselect(
                "Selecciona los IDs de las resistencias para el cálculo",
                available_ids,
                key=f"select_ids_{file_key}"
            )
            calculate_button = st.form_submit_button("Calcular")

            if calculate_button and selected_ids:
                indices = [int(id) - 1 for id in selected_ids]
                selected_values = [parse_resistance(resistor_values[i]['label']) for i in indices if parse_resistance(resistor_values[i]['label']) is not None]

                if selected_values:
                    if calculation_type == "Serie":
                        total_resistance = sum(selected_values)
                        st.write(f"Resistencia total en **serie**: **{format_resistance(total_resistance)}**")
                    else:  # Paralelo
                        try:
                            inverse_sum = sum(1 / val for val in selected_values)
                            total_resistance = 1 / inverse_sum
                            st.write(f"Resistencia total en **paralelo**: **{format_resistance(total_resistance)}**")
                        except ZeroDivisionError:
                            st.error("No se puede calcular la resistencia en paralelo con resistencias de cero ohmios.")
                else:
                    st.warning("No se seleccionaron valores de resistencia válidos para el cálculo.")
    else:
        st.write("No se detectaron resistencias.")