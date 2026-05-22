"""
Funciones auxiliares para la app Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st


# ------------------------------------------------------------------
# Pre-procesamiento clásico
# ------------------------------------------------------------------
def aplicar_preprocesamiento(imagen_rgb: np.ndarray):
    """Devuelve (gris, binarizada_otsu, sin_ruido) — útiles para enseñanza."""
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoise = cv2.fastNlMeansDenoising(gris, None, 10, 7, 21)
    return gris, binarizada, denoise


# ------------------------------------------------------------------
# EasyOCR (carga cacheada)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_easyocr(idiomas: List[str]):
    """Carga EasyOCR una sola vez por sesión."""
    import easyocr

    return easyocr.Reader(idiomas, gpu=False)


# ------------------------------------------------------------------
# Dibujo de cajas
# ------------------------------------------------------------------
def dibujar_cajas(imagen: np.ndarray, resultados) -> np.ndarray:
    """Dibuja cajas envolventes + texto sobre la imagen original."""
    for caja, texto, conf in resultados:
        pts = np.array(caja, dtype=np.int32)
        cv2.polylines(imagen, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        x, y = pts[0]
        cv2.putText(
            imagen,
            f"{texto} ({conf:.0%})",
            (int(x), max(int(y) - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )
    return imagen


# ------------------------------------------------------------------
# Modelo CNN+CTC propio (opcional)
# ------------------------------------------------------------------
PESOS_PATH = Path(__file__).parent.parent / "modelo_personalizado" / "pesos.h5"


@st.cache_resource(show_spinner=False)
def intentar_cargar_modelo_propio():
    """Intenta cargar el modelo CNN+CTC entrenado; devuelve None si no existe."""
    if not PESOS_PATH.exists():
        return None
    try:
        import sys

        sys.path.append(str(PESOS_PATH.parent))
        from modelo import construir_modelo_inferencia  # type: ignore

        modelo = construir_modelo_inferencia()
        modelo.load_weights(str(PESOS_PATH))
        return modelo
    except Exception as e:  # pragma: no cover
        st.error(f"Error al cargar modelo propio: {e}")
        return None


def predecir_con_modelo_propio(modelo, imagen_rgb: np.ndarray) -> str:
    """Predice una palabra con el modelo CNN+CTC entrenado en EMNIST."""
    import sys

    sys.path.append(str(PESOS_PATH.parent))
    from decodificador_ctc import decodificar_greedy  # type: ignore

    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    # Normaliza al tamaño esperado por el modelo (32x128 por convención)
    redimensionada = cv2.resize(gris, (128, 32))
    normalizada = redimensionada.astype("float32") / 255.0
    entrada = normalizada[np.newaxis, ..., np.newaxis]

    prediccion = modelo.predict(entrada, verbose=0)
    return decodificar_greedy(prediccion)
