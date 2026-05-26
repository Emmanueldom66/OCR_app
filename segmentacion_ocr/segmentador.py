# segmentacion_ocr/segmentador.py
"""
Segmentador de palabras basado en OpenCV.
Extrae caracteres individuales de una imagen de palabra usando procesamiento de contornos.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def preprocesar_para_segmentacion(
    imagen_rgb: np.ndarray, 
    invertir: bool = True,
    umbral_otsu: bool = True,
    aplicar_denoise: bool = True
) -> np.ndarray:
    """
    Prepara la imagen para la segmentación de caracteres.
    """
    # Convertir a escala de grises
    if len(imagen_rgb.shape) == 3:
        gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gris = imagen_rgb.copy()
    
    # Eliminar ruido
    if aplicar_denoise:
        gris = cv2.fastNlMeansDenoising(gris, None, 10, 7, 21)
    
    # Binarización
    if umbral_otsu:
        _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binarizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    
    # Invertir para que los caracteres sean blancos sobre fondo negro
    if invertir:
        binarizada = cv2.bitwise_not(binarizada)
    
    return binarizada


def segmentar_palabra(
    imagen_binaria: np.ndarray,
    altura_min: int = 10,
    anchura_min: int = 5,
    padding: int = 2
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extrae caracteres individuales mediante detección de contornos.
    
    Args:
        imagen_binaria: Imagen binarizada con caracteres en blanco y fondo negro
        altura_min: Altura mínima de un carácter (en píxeles)
        anchura_min: Anchura mínima de un carácter (en píxeles)
        padding: Espacio de relleno alrededor de cada carácter
    
    Returns:
        Lista de tuplas (imagen_recortada, (x, y, w, h)) ordenadas de izquierda a derecha
    """
    # Encontrar contornos
    contornos, _ = cv2.findContours(
        imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filtrar contornos por tamaño y obtener bounding boxes
    bounding_boxes = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= altura_min and w >= anchura_min:
            # Añadir padding
            bounding_boxes.append((x - padding, y - padding, w + 2*padding, h + 2*padding))
    
    # Ordenar de izquierda a derecha (por coordenada x)
    bounding_boxes.sort(key=lambda bb: bb[0])
    
    # Extraer y redimensionar cada carácter
    caracteres = []
    alto_original = imagen_binaria.shape[0]
    ancho_original = imagen_binaria.shape[1]
    
    for (x, y, w, h) in bounding_boxes:
        # Asegurar que las coordenadas no se salgan de la imagen
        x = max(0, x)
        y = max(0, y)
        w = min(w, ancho_original - x)
        h = min(h, alto_original - y)
        
        # Recortar carácter
        caracter = imagen_binaria[y:y+h, x:x+w]
        
        # Redimensionar a 28x28 (tamaño esperado por EMNIST)
        caracter_rs = cv2.resize(caracter, (28, 28))
        
        # Normalizar a [0, 1]
        caracter_rs = caracter_rs.astype("float32") / 255.0
        
        # Aplicar corrección de orientación (EMNIST está rotada)
        caracter_rs = np.rot90(caracter_rs, k=1)
        caracter_rs = np.rot90(caracter_rs, k=1)
        #caracter_rs = np.fliplr(caracter_rs)
        
        caracteres.append((caracter_rs, (x, y, w, h)))
    
    return caracteres


def segmentar_y_visualizar(imagen_rgb: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Segmenta la imagen y devuelve los caracteres junto con la imagen anotada.
    """
    # Preprocesar
    binaria = preprocesar_para_segmentacion(imagen_rgb)
    
    # Segmentar
    caracteres = segmentar_palabra(binaria)
    
    # Crear imagen con bounding boxes
    imagen_anotada = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
    for i, (caracter, (x, y, w, h)) in enumerate(caracteres):
        cv2.rectangle(imagen_anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            imagen_anotada, str(i+1), (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
    
    return [caract for caract, _ in caracteres], imagen_anotada


# ------------------------------------------------------------------
# Corrección de inclinación (deskew)
# ------------------------------------------------------------------
def corregir_inclinacion(imagen_rgb: np.ndarray) -> np.ndarray:
    """Corrige la inclinación de una imagen de texto."""
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    
    # Binarizar
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Encontrar coordenadas de píxeles blancos (texto)
    coords = np.column_stack(np.where(binaria > 0))
    if len(coords) < 10:
        return imagen_rgb
    
    # Calcular ángulo con transformada de Hough
    lineas = cv2.HoughLinesP(
        binaria, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lineas is not None:
        angulos = []
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            if x2 != x1:
                angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angulos.append(angulo)
        
        if angulos:
            angulo_med = np.median(angulos)
            if abs(angulo_med) > 0.5:
                # Rotar la imagen
                centro = (binaria.shape[1] // 2, binaria.shape[0] // 2)
                matriz = cv2.getRotationMatrix2D(centro, angulo_med, 1.0)
                imagen_corregida = cv2.warpAffine(
                    imagen_rgb, matriz, (binaria.shape[1], binaria.shape[0]),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                return imagen_corregida
    
    return imagen_rgb


if __name__ == "__main__":
    # Prueba básica
    import sys
    from pathlib import Path
    
    # Cargar imagen de ejemplo
    test_img_path = Path(__file__).parent.parent / "samples" / "palabra.png"
    if test_img_path.exists():
        img = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Segmentar
        caracteres, img_anotada = segmentar_y_visualizar(img_rgb)
        
        print(f"Se encontraron {len(caracteres)} caracteres")
        
        # Mostrar resultados
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original")
        axes[1].imshow(img_anotada)
        axes[1].set_title("Caracteres detectados")
        plt.show()