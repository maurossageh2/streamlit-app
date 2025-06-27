# Detección y Clasificación de Resistencias

Esta es una aplicación web desarrollada con **Streamlit** que permite detectar resistencias electrónicas en imágenes y clasificar sus valores de resistencia basándose en las bandas de color. Utiliza modelos de aprendizaje profundo de la biblioteca **Ultralytics YOLO** para la detección y clasificación, y ofrece funcionalidades adicionales como el cálculo de resistencias en serie o en paralelo.

## Requisitos

Para ejecutar esta aplicación, asegúrate de tener instalado lo siguiente:

- Python 3.8 o superior
- Las siguientes bibliotecas de Python:
  ```bash
  pip install streamlit opencv-python numpy pillow ultralytics pandas
  ```
- Modelos preentrenados YOLO:
  - `resistor_yolov8_v2.pt`: Modelo para detectar resistencias en la imagen.
  - `resistor_bands_yolov8.pt`: Modelo para clasificar los valores de las resistencias según las bandas de color.
  - Coloca estos modelos en la carpeta `./models/` relativa al directorio del script.

## Instalación

1. Clona este repositorio o descarga el código fuente.
2. Instala las dependencias ejecutando:
   ```bash
   pip install -r requirements.txt
   ```
   (Crea un archivo `requirements.txt` con las bibliotecas mencionadas si no está presente).
3. Asegúrate de que los modelos YOLO (`resistor_yolov8_v2.pt` y `resistor_bands_yolov8.pt`) estén en la carpeta `./models/`.
4. Ejecuta la aplicación con:
   ```bash
   streamlit run app.py
   ```
   donde `app.py` es el nombre del archivo que contiene el código.

## Uso

1. **Cargar Imágenes**: 
   - Accede a la aplicación en tu navegador (normalmente en `http://localhost:8501`).
   - Usa el componente de carga de archivos para subir una o varias imágenes (formatos compatibles: JPG, PNG, JPEG).
   - Las imágenes deben contener resistencias electrónicas visibles con bandas de color claras.

2. **Consejos para Mejores Resultados**:
   - **Iluminación**: Usa una iluminación brillante y uniforme para evitar sombras.
   - **Calidad de Imagen**: Sube imágenes de alta resolución para una mejor detección de colores.
   - **Distancia**: Mantén una distancia razonable entre las resistencias para evitar superposiciones.
   - **Fondo**: Usa un fondo limpio y simple para mejorar la precisión.

3. **Resultados**:
   - La aplicación detectará las resistencias en cada imagen y mostrará:
     - La imagen original.
     - Una imagen anotada con cuadros delimitadores y valores de resistencia.
     - Una tabla con los IDs de las resistencias, sus valores estimados y la confianza de la predicción.
   - Si no se detectan resistencias, se mostrará un mensaje indicando esto.

4. **Cálculo de Resistencias**:
   - Selecciona el tipo de cálculo (serie o paralelo) desde el formulario.
   - Elige los IDs de las resistencias que deseas incluir en el cálculo.
   - Haz clic en "Calcular" para obtener la resistencia total equivalente, formateada en ohmios, kiloohmios (kΩ) o megaohmios (MΩ).

## Características

- **Detección de Resistencias**: Utiliza un modelo YOLO para identificar resistencias en imágenes con una confianza mínima de 0.15.
- **Clasificación de Bandas de Color**: Un segundo modelo YOLO clasifica las bandas de color para determinar el valor de la resistencia.
- **Interfaz Intuitiva**: Muestra imágenes originales y anotadas lado a lado, junto con una tabla de resultados.
- **Cálculo de Resistencias**: Calcula la resistencia total en configuraciones en serie o en paralelo.
- **Soporte Multi-Imagen**: Procesa múltiples imágenes simultáneamente, con resultados organizados por archivo.
- **Optimización de Recursos**: Utiliza `@st.cache_resource` para cargar los modelos una sola vez y mejorar el rendimiento.

## Estructura del Código

- **Configuración Inicial**:
  - Configura Streamlit con un diseño ancho y un título.
  - Define funciones auxiliares para parsear y formatear valores de resistencia (por ejemplo, "4.7k" a 4700 ohmios).
- **Carga de Modelos**:
  - Carga dos modelos YOLO para detección y clasificación.
- **Procesamiento de Imágenes**:
  - Lee imágenes cargadas, las convierte a formato RGB y detecta resistencias.
  - Extrae regiones de interés (ROIs) y clasifica los valores de resistencia.
  - Anota las imágenes con cuadros delimitadores, IDs y valores de resistencia.
- **Resultados y Cálculos**:
  - Muestra imágenes originales y anotadas, junto con una tabla de resultados.
  - Proporciona un formulario para calcular resistencias en serie o en paralelo.
- **Gestión de Estado**:
  - Usa `st.session_state` para almacenar los resultados de cada imagen procesada.

## Limitaciones

- **Precisión de Detección**: Depende de la calidad de la imagen, la iluminación y la claridad de las bandas de color.
- **Valores Desconocidos**: Si el modelo de clasificación no puede identificar las bandas de color, el valor se marcará como "Desconocido".
- **Cálculos**: Los cálculos en paralelo no son posibles si alguna resistencia tiene un valor de cero ohmios.
- **Fuentes**: La aplicación intenta usar fuentes como Arial o DejaVuSans para las anotaciones. Si no están disponibles, usa una fuente por defecto, lo que puede afectar la legibilidad.

## Solución de Problemas

- **Error al cargar modelos**: Asegúrate de que los archivos `.pt` estén en la carpeta `./models/` y que los nombres coincidan exactamente.
- **No se detectan resistencias**: Verifica la calidad de la imagen y asegúrate de seguir los consejos para mejores resultados.
- **Errores de fuente**: Si las anotaciones no se ven bien, instala las fuentes Arial o DejaVuSans en tu sistema.
- **Problemas de rendimiento**: Si la aplicación es lenta, reduce el tamaño de las imágenes o usa menos imágenes a la vez.
