# Proyecto IA — Sistema OCR Educativo

**Reconocimiento Óptico de Caracteres (OCR) con Aprendizaje Profundo**

Autor: Emmanuel Domínguez Osio & Santiago Cruz Plaza
Curso: Inteligencia Artificial — Licenciatura en Mecatrónica  
Fecha: Mayo 2026

---

## 1. Objetivo

Este repositorio presenta un sistema OCR didáctico que combina **tres enfoques**:

1. **Pipeline pre‑entrenado** ([EasyOCR](https://github.com/JaidedAI/EasyOCR)) – detección + reconocimiento de texto en escenas reales.  
2. **Modelo CNN+CTC entrenado desde cero** – red CRNN+CTC entrenada sobre pseudo‑palabras EMNIST.  
3. **Segmentación por contornos + clasificador CNN** – extrae caracteres individuales y los clasifica con una CNN entrenada en EMNIST.

El objetivo es educativo: comparar distintos paradigmas (secuencial vs. segmentación) y entender las etapas internas del OCR.

---

## 2. Arquitectura del sistema

```
┌────────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│  Imagen de     │ -> │ Pre‑           │ -> │ Detección de    │ -> │ Reconocimiento │
│  entrada       │    │ procesamiento  │    │ texto (DBNet/   │    │ (CRNN+CTC /    │
│                │    │ (gris, binari- │    │  CRAFT)         │    │  Transformer)  │
│                │    │  zación, etc.) │    │                 │    │                │
└────────────────┘    └────────────────┘    └─────────────────┘    └────────────────┘
                                                                            │
                                                                            v
                                                                    ┌────────────────┐
                                                                    │ Post‑procesado │
                                                                    │ (corrección,   │
                                                                    │  diccionario)  │
                                                                    └────────────────┘
```

- **Pre‑procesamiento**: escala de grises, binarización (Otsu), denoising, corrección de inclinación (`skew correction`).
- **Detección**: localiza las regiones de texto (cajas envolventes). EasyOCR utiliza [CRAFT](https://github.com/clovaai/CRAFT-pytorch); PaddleOCR utiliza DBNet.
- **Reconocimiento**: transforma cada recorte en cadena de texto. La red estándar es **CRNN+CTC** ([Shi et al., 2015](https://arxiv.org/abs/1507.05717)).
- **Post‑procesamiento**: aplica diccionarios o modelos de lenguaje para corregir errores.


---

## 3. Estructura del repositorio

```
ocr_proyecto_ia/
├── app/                         # Aplicación Streamlit
│   ├── streamlit_app.py
│   └── utils.py
├── modelo_personalizado/        # CRNN+CTC entrenado desde cero
│   ├── modelo.py
│   ├── entrenar.py
│   ├── inferencia.py
│   └── decodificador_ctc.py
├── segmentacion_ocr/            # Nuevo enfoque: segmentación + CNN
│   ├── README.md                # Documentación específica
│   ├── clasificador_emnist.py
│   ├── segmentador.py
│   ├── pipeline_segmentacion.py
│   └── clasificador_pesos.weights.h5
├── notebooks/
│   └── 01_emnist_cnn_ctc.ipynb
├── samples/                     # Imágenes de ejemplo
├── docs/
│   ├── Reporte_OCR.pdf
│   └── Presentacion_OCR.pptx
├── assets/                      # Diagramas y figuras
├── requirements.txt
└── README.md
```

---

## 4. Instalación

```bash
git clone <tu_repo>
cd ocr_proyecto_ia
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**Dependencias clave**: `easyocr`, `tensorflow`, `streamlit`, `opencv-python`, `numpy`, `matplotlib`, `tensorflow-datasets` (para EMNIST).

---

## 5. Uso de los tres enfoques

### a) Demo web (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

En la barra lateral puedes seleccionar:
- **EasyOCR** (siempre activo).  
- **Modelo CNN+CTC propio** (marcar casilla, requiere pesos entrenados).  
- **Segmentación + CNN** (marcar casilla, requiere clasificador entrenado).

La app muestra:
- Imagen original y pre‑procesada.
- Cajas detectadas (EasyOCR).
- Texto reconocido y confianza.
- Para segmentación: pasos intermedios y confianza por carácter.

### b) Entrenar el modelo CRNN+CTC propio

```bash
python modelo_personalizado/entrenar.py --epochs 10 --batch-size 64
```
Los pesos se guardan en `modelo_personalizado/pesos.h5`.

### c) Entrenar el clasificador de segmentación (CNN)

```bash
cd segmentacion_ocr
python clasificador_emnist.py --epochs 20 --batch-size 64
```
Los pesos se guardan en `segmentacion_ocr/clasificador_pesos.weights.h5`.

### d) Inferencia con el modelo propio (línea de comandos)

```bash
python modelo_personalizado/inferencia.py --imagen samples/palabra.png
```

### e) Inferencia con segmentación (línea de comandos)

```bash
cd segmentacion_ocr
python pipeline_segmentacion.py --imagen ../samples/palabra.png
```

---

## 6. Comparativa de los tres enfoques

| Método                    | Tipo de entrenamiento            | Longitud variable | Interpretabilidad | Tamaño modelo | Precisión (EMNIST) |
|---------------------------|----------------------------------|-------------------|-------------------|---------------|--------------------|
| EasyOCR (pre‑entrenado)   | Masivo (multilingüe)             | Sí                | Baja              | ~64 MB        | No aplicable       |
| CRNN+CTC propio           | Pseudo‑palabras EMNIST (secuencias) | No (fija)      | Media             | ~17 MB        | 78% (palabra)      |
| Segmentación + CNN        | Caracteres EMNIST                | Sí                | Alta              | ~5 MB         | 99% (carácter)     |

---

## 7. Conjuntos de datos y repositorios externos

| Dataset       | Uso                                      | Enlace |
|---------------|------------------------------------------|--------|
| EMNIST        | Entrenamiento clasificador CNN y CRNN+CTC | [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) |
| MNIST         | Fallback                                 | [Lecun](http://yann.lecun.com/exdb/mnist/) |
| IAM           | Referencia (no usado directamente)       | [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) |
| ICDAR‑SROIE   | Referencia                               | [SROIE](https://rrc.cvc.uab.es/?ch=13) |

Repositorios comparados: [EasyOCR](https://github.com/JaidedAI/EasyOCR), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr), [docTR](https://github.com/mindee/doctr), [MMOCR](https://github.com/open-mmlab/mmocr), [CRAFT](https://github.com/clovaai/CRAFT-pytorch).

---

## 8. Licencia

Código bajo licencia MIT (fines educativos). Los modelos y datasets de terceros mantienen sus propias licencias.

---

## 9. Citación

> Cruz Plaza, S. & Domínguez Osio, E. (2026). *Sistema OCR Educativo con CNN+CTC y EasyOCR*. Proyecto de Inteligencia Artificial, Licenciatura en Ingeniería en Mecatrónica.
