# Proyecto IA — Sistema OCR Educativo

**Reconocimiento Óptico de Caracteres (OCR) con Aprendizaje Profundo**

Autor: Emmanuel Domínguez Osio
Programa: Ingeniería en Mecatrónica / Sistemas de Control
Curso: Inteligencia Artificial (Licenciatura)

---

## 1. Objetivo

Este repositorio presenta un sistema OCR didáctico que combina:

1. **Un pipeline pre‑entrenado** ([EasyOCR](https://github.com/JaidedAI/EasyOCR)) para detección + reconocimiento de texto en escenas reales.
2. **Un modelo CNN+CTC entrenado desde cero** sobre el dataset EMNIST/MNIST, para explicar paso a paso cómo aprende una red neuronal a leer caracteres.
3. **Una interfaz web** en [Streamlit](https://streamlit.io/) donde el usuario sube una imagen y observa el resultado de ambos enfoques.

El objetivo es educativo: permitir al estudiante comparar un sistema de producción contra un modelo propio y comprender las etapas internas (pre‑procesamiento → detección → reconocimiento → post‑procesamiento).

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
├── app/                      # Aplicación Streamlit (demo interactiva)
│   ├── streamlit_app.py
│   └── utils.py
├── modelo_personalizado/     # CNN+CTC entrenado desde cero
│   ├── modelo.py             # Definición de la red
│   ├── entrenar.py           # Script de entrenamiento
│   ├── inferencia.py         # Predicción sobre imágenes
│   └── decodificador_ctc.py  # Decodificador greedy / beam
├── notebooks/
│   └── 01_emnist_cnn_ctc.ipynb   # Notebook didáctico paso a paso
├── samples/                  # Imágenes de ejemplo
├── docs/                     # Reporte y presentación
│   ├── Reporte_OCR.pdf
│   └── Presentacion_OCR.pptx
├── assets/                   # Diagramas, figuras
├── requirements.txt
└── README.md
```

---

## 4. Instalación

```bash
# 1. Clonar y crear entorno
git clone <tu_repo>
cd ocr_proyecto_ia
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

Dependencias principales:

| Paquete       | Uso                                  |
|---------------|--------------------------------------|
| `easyocr`     | Pipeline pre‑entrenado (CRAFT+CRNN)  |
| `torch`       | Backend para EasyOCR y modelo propio |
| `tensorflow`  | Entrenamiento del modelo CNN+CTC     |
| `streamlit`   | Interfaz web                         |
| `opencv-python` | Pre‑procesamiento de imágenes      |
| `numpy`, `pillow`, `matplotlib` | Utilidades             |

---

## 5. Uso rápido

### a) Demo web (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Sube una imagen con texto y observa:
- Las cajas envolventes detectadas por EasyOCR.
- El texto transcrito.
- (Opcional) La predicción de tu modelo CNN+CTC entrenado sobre EMNIST.

### b) Entrenar el modelo propio

```bash
python modelo_personalizado/entrenar.py --epochs 10 --batch-size 64
```

El script descarga EMNIST automáticamente y guarda los pesos en `modelo_personalizado/pesos.h5`.

### c) Inferencia con el modelo propio

```bash
python modelo_personalizado/inferencia.py --imagen samples/palabra.png
```

---

## 6. Repositorios y datasets de referencia

### Repositorios OCR estudiados

| Repositorio | Enfoque | URL |
|-------------|---------|-----|
| Tesseract OCR | OCR clásico + LSTM | https://github.com/tesseract-ocr/tesseract |
| EasyOCR | CRAFT + CRNN | https://github.com/JaidedAI/EasyOCR |
| PaddleOCR | DBNet + CRNN/SVTR | https://github.com/PaddlePaddle/PaddleOCR |
| docTR | Detección + reconocimiento documental | https://github.com/mindee/doctr |
| TrOCR | Transformer end‑to‑end | https://github.com/microsoft/unilm/tree/master/trocr |
| CRAFT | Detección a nivel de carácter | https://github.com/clovaai/CRAFT-pytorch |
| Surya | OCR multi‑idioma con análisis de layout | https://github.com/VikParuchuri/surya |
| MMOCR | Toolkit OpenMMLab | https://github.com/open-mmlab/mmocr |

### Datasets utilizados / mencionados

| Dataset | Tipo | Tamaño | Fuente |
|---------|------|--------|--------|
| MNIST | Dígitos manuscritos | 70 000 | http://yann.lecun.com/exdb/mnist/ |
| EMNIST | Caracteres alfanuméricos manuscritos | 814 255 | https://www.nist.gov/itl/products-and-services/emnist-dataset |
| IAM Handwriting DB | Texto manuscrito en inglés | 13 353 líneas | https://fki.tic.heia-fr.ch/databases/iam-handwriting-database |
| ICDAR‑SROIE | Recibos escaneados | 1 000 imágenes | https://rrc.cvc.uab.es/?ch=13 |
| Synth90k / MJSynth | Palabras sintéticas | 9 millones | https://www.robots.ox.ac.uk/~vgg/data/text/ |
| SynthText | Texto en escenas sintéticas | 800 000 imágenes | https://www.robots.ox.ac.uk/~vgg/data/scenetext/ |

---

## 7. Licencia

Código liberado bajo licencia MIT con fines educativos. Los modelos y datasets de terceros mantienen sus licencias originales.

---

## 8. Cómo citar este proyecto

> Domínguez Osio, E. (2026). *Sistema OCR Educativo con CNN+CTC y EasyOCR*. Proyecto de Inteligencia Artificial, Licenciatura en Ingeniería en Mecatrónica.
