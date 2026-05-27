"""
Aplicación Streamlit — Demo OCR educativa
==========================================

Esta interfaz permite cargar una imagen y comparar dos enfoques OCR:

1. Pipeline pre-entrenado (EasyOCR): detección + reconocimiento de texto.
2. Modelo CNN+CTC propio entrenado sobre EMNIST (opcional).

Autor: Emmanuel Domínguez Osio
Proyecto IA - Licenciatura en Mecatrónica
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils import (
    aplicar_preprocesamiento,
    cargar_easyocr,
    dibujar_cajas,
    intentar_cargar_modelo_propio,
    predecir_con_modelo_propio,
)


# ------------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="OCR Educativo - Proyecto IA",
    page_icon="🔤",
    layout="wide",
)

st.title("🔤 Sistema OCR Educativo")
st.markdown(
    """
    Proyecto académico de **Inteligencia Artificial** que compara dos enfoques
    para Reconocimiento Óptico de Caracteres:

    - **Pipeline pre-entrenado** ([EasyOCR](https://github.com/JaidedAI/EasyOCR)):
      detección con CRAFT + reconocimiento con CRNN+CTC.
    - **Modelo CNN+CTC propio**: red neuronal entrenada desde cero sobre
      EMNIST para fines didácticos.
    """
)

# ------------------------------------------------------------------
# Barra lateral - controles
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    idioma = st.multiselect(
        "Idiomas a reconocer",
        options=["es", "en", "fr", "de", "pt", "it"],
        default=["es", "en"],
        help="EasyOCR soporta más de 80 idiomas.",
    )

    mostrar_preproc = st.checkbox("Mostrar etapas de pre-procesamiento", value=True)
    mostrar_cajas = st.checkbox("Dibujar cajas de detección", value=True)
    usar_modelo_propio = st.checkbox(
        "Probar también el modelo CNN+CTC propio", value=False,
        help="Requiere que los pesos estén en modelo_personalizado/pesos.weights.h5",
    )

    st.markdown("---")
    st.caption(
        "Pipeline OCR estándar:\n"
        "1. Pre-procesamiento\n"
        "2. Detección de texto\n"
        "3. Reconocimiento\n"
        "4. Post-procesamiento"
    )

# ------------------------------------------------------------------
# Carga de imagen
# ------------------------------------------------------------------
archivo = st.file_uploader(
    "Sube una imagen (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
)

if archivo is None:
    st.info("👆 Sube una imagen para comenzar. También puedes probar con los ejemplos en la carpeta `samples/`.")
    st.stop()

imagen_pil = Image.open(archivo).convert("RGB")
imagen_np = np.array(imagen_pil)

# ------------------------------------------------------------------
# Pre-procesamiento
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Imagen original")
    st.image(imagen_pil, use_column_width=True)

if mostrar_preproc:
    gris, binarizada, denoise = aplicar_preprocesamiento(imagen_np)
    with col2:
        st.subheader("🔧 Pre-procesamiento")
        tabs = st.tabs(["Escala de grises", "Binarización (Otsu)", "Sin ruido"])
        with tabs[0]:
            st.image(gris, use_column_width=True, clamp=True)
        with tabs[1]:
            st.image(binarizada, use_column_width=True, clamp=True)
        with tabs[2]:
            st.image(denoise, use_column_width=True, clamp=True)

# ------------------------------------------------------------------
# Visualización compartida (EasyOCR y modelo propio)
# ------------------------------------------------------------------
def _mostrar_resultados_ocr(titulo: str, resultados, imagen_base, tiempo: float):
    """Métricas, imagen anotada y texto — mismo layout para ambos modelos."""
    col_img, col_met = st.columns([2, 1])
    with col_img:
        if mostrar_cajas and resultados:
            imagen_anotada = dibujar_cajas(imagen_base.copy(), resultados)
            st.image(imagen_anotada, use_column_width=True)
        else:
            st.image(imagen_pil, use_column_width=True)

    with col_met:
        st.metric("Tiempo de inferencia", f"{tiempo:.2f} s")
        st.metric("Regiones detectadas", len(resultados))
        if resultados:
            conf_promedio = np.mean([r[2] for r in resultados])
            st.metric("Confianza promedio", f"{conf_promedio:.2%}")

    st.subheader("📝 Texto reconocido")
    if not resultados:
        st.warning("No se detectó texto en la imagen.")
        return

    for i, (_caja, texto, conf) in enumerate(resultados, 1):
        st.markdown(f"**{i}.** `{texto}`  &nbsp;&nbsp; *(confianza: {conf:.2%})*")

    texto_completo = "\n".join(r[1] for r in resultados)
    st.text_area("Texto plano", texto_completo, height=120, key=f"texto_{titulo}")
    st.download_button(
        "💾 Descargar como .txt",
        data=texto_completo.encode("utf-8"),
        file_name=f"texto_extraido_{titulo}.txt",
        mime="text/plain",
        key=f"descarga_{titulo}",
    )


# ------------------------------------------------------------------
# Detección + reconocimiento con EasyOCR
# ------------------------------------------------------------------
st.markdown("---")
st.header("🤖 Resultado con EasyOCR (pre-entrenado)")

with st.spinner("Cargando modelo EasyOCR (primera vez puede tardar)..."):
    reader = cargar_easyocr(idioma or ["en"])

t0 = time.time()
resultados = reader.readtext(imagen_np)
t1 = time.time()

_mostrar_resultados_ocr("easyocr", resultados, imagen_np, t1 - t0)

# ------------------------------------------------------------------
# Modelo CNN+CTC propio (opcional)
# ------------------------------------------------------------------
if usar_modelo_propio:
    st.markdown("---")
    st.header("🧠 Resultado con modelo CNN+CTC propio (EMNIST)")
    
    with st.spinner("Prediciendo con la CNN+CTC..."): 
        modelo = intentar_cargar_modelo_propio()
        
    t0_propio = time.time()
    resultados_propio = predecir_con_modelo_propio(modelo, imagen_np)
    t1_propio = time.time()
    _mostrar_resultados_ocr("propio", resultados_propio, imagen_np, t1_propio - t0_propio)
    st.caption(
        "Nota: el modelo propio fue entrenado sobre caracteres aislados "
        "(EMNIST). Trata la imagen completa como una sola región de texto."
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Proyecto desarrollado con fines educativos · "
    "Emmanuel Domínguez Osio · "
    "Curso de Inteligencia Artificial · "
    "Mayo 2026"
)
