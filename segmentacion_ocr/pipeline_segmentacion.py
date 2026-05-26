# segmentacion_ocr/pipeline_segmentacion.py
"""
Pipeline OCR completo basado en segmentación por contornos + clasificación CNN.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# Importar módulos locales
from segmentacion_ocr.segmentador import (
    preprocesar_para_segmentacion,
    segmentar_palabra,
    segmentar_y_visualizar,
    corregir_inclinacion
)
from segmentacion_ocr.clasificador_emnist import (
    ALFABETO, TAM_VOCAB,
    construir_clasificador,
    predecir_imagen
)


# ------------------------------------------------------------------
# Carga del modelo clasificador
# ------------------------------------------------------------------
def cargar_clasificador(ruta_pesos: str = None) -> tf.keras.Model:
    """Carga el clasificador CNN entrenado."""
    if ruta_pesos is None:
        ruta_pesos = Path(__file__).parent / "clasificador_pesos.weights.h5"
    
    modelo = construir_clasificador()
    if Path(ruta_pesos).exists():
        modelo.load_weights(str(ruta_pesos))
        print(f"✅ Clasificador cargado desde {ruta_pesos}")
    else:
        print(f"⚠️  No se encontraron pesos en {ruta_pesos}")
        print("   Ejecuta: python clasificador_emnist.py --epochs 15")
    
    return modelo


# ------------------------------------------------------------------
# Pipeline completo
# ------------------------------------------------------------------
def reconocer_palabra_segmentacion(
    modelo: tf.keras.Model,
    imagen_rgb: np.ndarray,
    corregir_inclinacion_img: bool = True,
    visualizar: bool = False
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Reconocimiento de palabra mediante segmentación + clasificación.
    
    Returns:
        texto: palabra reconocida
        detalles: lista de tuplas (carácter predicho, confianza) para cada segmento
    """
    # Preprocesar
    if corregir_inclinacion_img:
        imagen_rgb = corregir_inclinacion(imagen_rgb)
    
    # Binarizar para segmentación
    binaria = preprocesar_para_segmentacion(imagen_rgb)
    
    # Segmentar palabra en caracteres
    caracteres = segmentar_palabra(binaria)
    
    # Clasificar cada carácter
    texto = []
    detalles = []
    for caracter_img, bb in caracteres:
        # caracter_img ya está en formato 28x28 normalizado
        pred_char, probs = predecir_imagen(modelo, caracter_img)
        confianza = np.max(probs)
        texto.append(pred_char)
        detalles.append((pred_char, confianza))
    
    return "".join(texto), detalles


def reconocer_imagen_completa(
    modelo: tf.keras.Model,
    imagen_rgb: np.ndarray,
    corregir_inclinacion_img: bool = True
) -> List[Tuple[str, List[Tuple[str, float]], np.ndarray]]:
    """
    Reconoce múltiples palabras en una imagen.
    
    Nota: Para imágenes con múltiples líneas, se necesita un detector de líneas.
    Esta implementación asume que la imagen contiene una sola palabra.
    """
    texto, detalles = reconocer_palabra_segmentacion(
        modelo, imagen_rgb, corregir_inclinacion_img, visualizar=False
    )
    return [(texto, detalles, imagen_rgb)]


# ------------------------------------------------------------------
# Interfaz para Streamlit
# ------------------------------------------------------------------
def mostrar_demo_segmentacion(modelo: tf.keras.Model, imagen_rgb: np.ndarray):
    """Muestra los pasos del pipeline de segmentación en Streamlit."""
    
    st.subheader("🔍 Paso 1: Preprocesamiento")
    binaria = preprocesar_para_segmentacion(imagen_rgb)
    st.image(binaria, caption="Imagen binarizada", use_container_width=True, clamp=True)
    
    st.subheader("✂️ Paso 2: Segmentación de caracteres")
    caracteres, img_anotada = segmentar_y_visualizar(imagen_rgb)
    st.image(img_anotada, caption="Caracteres detectados", use_container_width=True, clamp=True)
    
    st.subheader("🔤 Paso 3: Clasificación de caracteres")
    cols = st.columns(min(len(caracteres), 8))
    for i, (caracter, col) in enumerate(zip(caracteres, cols)):
        with col:
            st.image(caracter.reshape(28, 28), caption=f"Carácter {i+1}", use_container_width=True)
    
    st.subheader("📝 Paso 4: Resultado final")
    texto, detalles = reconocer_palabra_segmentacion(modelo, imagen_rgb)
    st.success(f"Palabra reconocida: **{texto}**")
    
    # Mostrar confianzas
    st.markdown("**Confianza por carácter:**")
    for i, (car, conf) in enumerate(detalles):
        st.markdown(f"- Carácter {i+1}: `{car}` ({conf:.2%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", type=str, help="Ruta a la imagen a reconocer")
    parser.add_argument("--pesos", type=str, default="clasificador_pesos.h5")
    args = parser.parse_args()
    
    modelo = cargar_clasificador(args.pesos)
    
    if args.imagen:
        img = cv2.imread(args.imagen)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            texto, detalles = reconocer_palabra_segmentacion(modelo, img_rgb)
            print(f"\n📝 Resultado: '{texto}'")
            for i, (car, conf) in enumerate(detalles):
                print(f"   Carácter {i+1}: '{car}' (confianza: {conf:.2%})")