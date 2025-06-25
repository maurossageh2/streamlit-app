import os
import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO

# Configuración inicial
INPUT_DIR = "input"  # Directorio con las 10 imágenes originales
OUTPUT_DIR = "output"  # Directorio para las imágenes generadas
CROPPED_DIR = "cropped_resistors"  # Directorio para resistencias recortadas
NUM_OUTPUT_IMAGES = 100  # Número de imágenes a generar
BG_WIDTH, BG_HEIGHT = 800, 600  # Tamaño del fondo
MAX_RESISTOR_WIDTH = 80  # Ancho máximo para resistencias redimensionadas
MODEL_PATH = "../models/resistor_yolov8_v2.pt"  # Ruta al modelo YOLO
BACKGROUND_IMAGE = "background.jpg"  # Ruta a la imagen de fondo
BRIGHTNESS_FACTOR = 0.95  # Factor para reducir el brillo de la resistencia (ajustable, 0.7 a 1.0)

# Cargar el modelo YOLO
model_resistor = YOLO(MODEL_PATH)

# Recortar resistencias de las imágenes originales
def crop_resistors():
    Path(CROPPED_DIR).mkdir(parents=True, exist_ok=True)
    resistor_images = []
    resistor_id = 0

    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"Directory {INPUT_DIR} does not exist.")
        return resistor_images

    # Procesar cada imagen en el directorio
    for image_path in input_path.glob('*.jpg'):
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convertir BGR a RGB para procesamiento
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Realizar detección
        results = model_resistor.predict(image_rgb)

        # Extraer ROIs
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Obtener bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                roi = image_rgb[y1:y2, x1:x2]
                # Convertir ROI de RGB a BGR para guardar
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(CROPPED_DIR, f"resistor_{resistor_id}.jpg")
                cv2.imwrite(output_path, roi_bgr)
                resistor_images.append(output_path)
                resistor_id += 1
                print(f'Saved resistor region {resistor_id} for {image_path.name} to {output_path}')

    return resistor_images

# Verificar si hay solapamiento entre bounding boxes
def check_overlap(new_box, used_boxes):
    x1, y1, x2, y2 = new_box
    for ux1, uy1, ux2, uy2 in used_boxes:
        if not (x2 <= ux1 or x1 >= ux2 or y2 <= uy1 or y1 >= uy2):
            return True
    return False

# Crear máscara para la resistencia excluyendo el fondo blanco
def create_resistor_mask(resistor):
    # Convertir a escala de grises
    gray = cv2.cvtColor(resistor, cv2.COLOR_BGR2GRAY)
    
    # Umbralización para separar el fondo blanco (asumimos blanco puro o cercano)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Dilatar ligeramente la máscara para incluir bordes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Suavizar los bordes de la máscara
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Convertir a formato de 3 canales para Poisson blending
    mask_3c = cv2.merge([mask, mask, mask])
    return mask_3c

# Generar nuevas imágenes
def generate_new_images(resistor_images):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Cargar y redimensionar la imagen de fondo
    bg_image = cv2.imread(BACKGROUND_IMAGE)
    if bg_image is None:
        print(f"Failed to load background image: {BACKGROUND_IMAGE}")
        return
    bg_image = cv2.resize(bg_image, (BG_WIDTH, BG_HEIGHT))

    for i in range(NUM_OUTPUT_IMAGES):
        # Usar una copia de la imagen de fondo
        background = bg_image.copy()
        used_boxes = []
        
        # Número aleatorio de resistencias entre 1 y 20
        num_resistors = random.randint(1, min(20, len(resistor_images)))
        selected_resistors = random.sample(resistor_images, num_resistors)
        
        for resistor_path in selected_resistors:
            resistor = cv2.imread(resistor_path)
            if resistor is None:
                print(f"Failed to load resistor image: {resistor_path}")
                continue
            
            # Obtener dimensiones originales
            h_orig, w_orig = resistor.shape[:2]
            # Calcular relación de aspecto
            aspect_ratio = h_orig / w_orig
            # Redimensionar manteniendo la relación de aspecto
            new_width = min(MAX_RESISTOR_WIDTH, w_orig)
            new_height = int(new_width * aspect_ratio)
            
            # Redimensionar la resistencia
            resistor = cv2.resize(resistor, (new_width, new_height))
            h, w = resistor.shape[:2]
            
            # Verificar que las dimensiones sean válidas
            if w > BG_WIDTH or h > BG_HEIGHT:
                print(f"Resistor at {resistor_path} too large for background, skipping.")
                continue
            
            # Ajustar el brillo de la resistencia
            adjusted_resistor = cv2.convertScaleAbs(resistor, alpha=BRIGHTNESS_FACTOR, beta=0)
            
            max_x, max_y = BG_WIDTH - w, BG_HEIGHT - h
            
            attempts = 0
            max_attempts = 50
            placed = False
            
            while attempts < max_attempts and not placed:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                new_box = (x, y, x + w, y + h)
                
                if not check_overlap(new_box, used_boxes):
                    # Crear máscara para la resistencia
                    mask = create_resistor_mask(adjusted_resistor)
                    
                    # Asegurar que la máscara y la imagen tengan el mismo tamaño
                    if mask.shape[:2] != adjusted_resistor.shape[:2]:
                        print(f"Mask size mismatch for {resistor_path}, skipping.")
                        continue
                    
                    # Usar Poisson blending para fusionar la resistencia
                    try:
                        # Centro de la resistencia para seamlessClone
                        center = (x + w // 2, y + h // 2)
                        # Aplicar Poisson blending
                        blended = cv2.seamlessClone(
                            adjusted_resistor, 
                            background, 
                            mask, 
                            center, 
                            cv2.NORMAL_CLONE
                        )
                        # Actualizar el fondo con la imagen fusionada
                        background = blended
                    except cv2.error as e:
                        print(f"Error in seamlessClone for {resistor_path}: {e}")
                        continue
                    
                    used_boxes.append(new_box)
                    placed = True
                attempts += 1
        
        # Guardar la imagen generada
        output_path = os.path.join(OUTPUT_DIR, f"generated_{i}.jpg")
        cv2.imwrite(output_path, background)
        print(f"Saved generated image: {output_path}")

def main():
    # Paso 1: Recortar resistencias
    resistor_images = crop_resistors()
    print(f"Se recortaron {len(resistor_images)} resistencias.")
    
    # Paso 2: Generar nuevas imágenes
    generate_new_images(resistor_images)
    print(f"Se generaron {NUM_OUTPUT_IMAGES} imágenes nuevas.")

if __name__ == "__main__":
    main()