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
import sys
from pathlib import Path
# Añadir la raíz del proyecto al path para poder importar segmentacion_ocr
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        help="Requiere que los pesos estén en modelo_personalizado/pesos.h5",
    )
    usar_segmentacion = st.checkbox(
        "Probar también el enfoque de Segmentación + CNN", value=False,
        help="Segmenta caracteres por contornos y clasifica con CNN entrenada en EMNIST.\nRequiere entrenar primero el clasificador.",
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
# Detección + reconocimiento con EasyOCR
# ------------------------------------------------------------------
st.markdown("---")
st.header("🤖 Resultado con EasyOCR (pre-entrenado)")

with st.spinner("Cargando modelo EasyOCR (primera vez puede tardar)..."):
    reader = cargar_easyocr(idioma if idioma else ["en"])

t0 = time.time()
resultados = reader.readtext(imagen_np)
t1 = time.time()

col3, col4 = st.columns([2, 1])
with col3:
    if mostrar_cajas:
        imagen_anotada = dibujar_cajas(imagen_np.copy(), resultados)
        st.image(imagen_anotada, use_column_width=True)
    else:
        st.image(imagen_pil, use_column_width=True)

with col4:
    st.metric("Tiempo de inferencia", f"{t1 - t0:.2f} s")
    st.metric("Regiones detectadas", len(resultados))
    if resultados:
        conf_promedio = np.mean([r[2] for r in resultados])
        st.metric("Confianza promedio", f"{conf_promedio:.2%}")

st.subheader("📝 Texto reconocido")
if not resultados:
    st.warning("No se detectó texto en la imagen.")
else:
    for i, (caja, texto, conf) in enumerate(resultados, 1):
        st.markdown(
            f"**{i}.** `{texto}`  &nbsp;&nbsp; *(confianza: {conf:.2%})*"
        )

    texto_completo = "\n".join(r[1] for r in resultados)
    st.text_area("Texto plano", texto_completo, height=120)
    st.download_button(
        "💾 Descargar como .txt",
        data=texto_completo.encode("utf-8"),
        file_name="texto_extraido.txt",
        mime="text/plain",
    )

# ------------------------------------------------------------------
# Enfoque Segmentación + CNN (nuevo)
# ------------------------------------------------------------------
if usar_segmentacion:
    st.markdown("---")
    st.header("✂️ Resultado con Segmentación + CNN (nuevo enfoque)")

    try:
        from segmentacion_ocr.pipeline_segmentacion import (
            cargar_clasificador,
            reconocer_palabra_segmentacion,
        )
        from segmentacion_ocr.segmentador import segmentar_y_visualizar
        import cv2

        with st.spinner("Cargando clasificador CNN..."):
            clasificador = cargar_clasificador()

        if clasificador is None:
            st.warning(
                "No se encontró el clasificador entrenado. "
                "Ejecuta: `cd segmentacion_ocr && python clasificador_emnist.py --epochs 15`"
            )
        else:
            with st.spinner("Reconociendo mediante segmentación..."):
                texto, detalles = reconocer_palabra_segmentacion(clasificador, imagen_np)

            st.success(f"Palabra reconocida: **{texto}**")

            if detalles:
                st.markdown("**Confianza por carácter:**")
                cols = st.columns(len(detalles))
                for i, (car, conf) in enumerate(detalles):
                    with cols[i]:
                        st.metric(f"Carácter {i+1}", car, f"{conf:.0%}")

            with st.expander("Mostrar pasos intermedios de segmentación"):
                caracteres, img_anotada = segmentar_y_visualizar(imagen_np)
                st.image(img_anotada, caption="Caracteres detectados", use_container_width=True)
                if caracteres:
                    st.markdown("**Caracteres extraídos:**")
                    cols_car = st.columns(min(len(caracteres), 8))
                    for i, car_img in enumerate(caracteres):
                        with cols_car[i % len(cols_car)]:
                            img28 = car_img.reshape(28, 28)
                            img_rot = np.rot90(img28, 2)
                            st.image(img_rot, caption=f"Carácter {i+1}", use_container_width=True)
                else:
                    st.info("No se detectaron caracteres.")
    except ImportError as e:
        st.error(f"Error al importar módulo de segmentación: {e}. Asegúrate de que el directorio `segmentacion_ocr` existe en la raíz del proyecto.")

# ------------------------------------------------------------------
# Modelo CNN+CTC propio (opcional)
# ------------------------------------------------------------------
if usar_modelo_propio:
    st.markdown("---")
    st.header("🧠 Resultado con modelo CNN+CTC propio (EMNIST)")

    modelo = intentar_cargar_modelo_propio()
    if modelo is None:
        st.warning(
            "No se encontraron pesos entrenados en "
            "`modelo_personalizado/pesos.h5`. Ejecuta primero:\n\n"
            "`python modelo_personalizado/entrenar.py --epochs 10`"
        )
    else:
        with st.spinner("Prediciendo con la CNN+CTC..."):
            prediccion = predecir_con_modelo_propio(modelo, imagen_np)
        st.success(f"Predicción del modelo propio: **{prediccion}**")
        st.caption(
            "Nota: el modelo propio fue entrenado sobre caracteres aislados "
            "(EMNIST). Funciona mejor con palabras manuscritas cortas en "
            "fondo claro."
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
