# Guía rápida de uso — Sistema OCR Educativo

Esta guía complementa al [README](README.md) y al reporte académico.
Está pensada para que cualquier compañero(a) pueda reproducir el proyecto en
menos de 30 minutos.

---

## 1. Requisitos previos

- Python ≥ 3.10
- Sistema operativo: Linux, macOS o Windows 10/11
- Espacio en disco: ~3 GB (incluye modelos pre-entrenados de EasyOCR)
- (Opcional) GPU NVIDIA con CUDA para entrenar más rápido

---

## 2. Instalación paso a paso

```bash
# Clona el repositorio
git clone <tu_repo>
cd ocr_proyecto_ia

# Crea entorno virtual
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows PowerShell

# Instala dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** La primera vez que ejecutes EasyOCR descargará automáticamente los
> modelos CRAFT y CRNN (≈ 100 MB).

---

## 3. Ejecutar la demo web

```bash
streamlit run app/streamlit_app.py
```

Abre el navegador en <http://localhost:8501>.

Flujo de la demo:

1. Sube una imagen (PNG/JPG).
2. Observa las **etapas de pre-procesamiento** (gris, Otsu, denoise).
3. Mira las **cajas verdes** que detecta CRAFT y el **texto reconocido**.
4. Descarga el resultado como `.txt`.

---

## 4. Entrenar el modelo CNN+CTC propio

```bash
python modelo_personalizado/entrenar.py --epochs 10 --batch-size 64
```

- Descarga EMNIST automáticamente (con fallback a MNIST si tensorflow-datasets falla).
- Genera 20 000 pseudo-palabras de 3–8 caracteres.
- Guarda los pesos en `modelo_personalizado/pesos.h5`.

Después puedes marcar la casilla **"Probar también el modelo CNN+CTC propio"**
en la app Streamlit para verlo en acción.

---

## 5. Inferencia desde la línea de comandos

```bash
python modelo_personalizado/inferencia.py \
    --imagen samples/palabra.png \
    --metodo greedy
```

Opciones:

| Argumento  | Valores            | Descripción                             |
|------------|--------------------|-----------------------------------------|
| `--imagen` | ruta a PNG/JPG     | Imagen a transcribir                    |
| `--pesos`  | ruta a `.h5`       | Pesos entrenados (por defecto `pesos.h5`)|
| `--metodo` | `greedy` / `beam`  | Algoritmo de decodificación CTC         |

---

## 6. Explorar el notebook didáctico

```bash
jupyter notebook notebooks/01_emnist_cnn_ctc.ipynb
```

El notebook te lleva paso a paso a través del entrenamiento de la red CRNN+CTC
y la decodificación greedy. Ideal para preparar la exposición oral.

---

## 7. Posibles errores y soluciones

| Síntoma                                        | Causa probable                     | Solución                                            |
|------------------------------------------------|------------------------------------|-----------------------------------------------------|
| `ImportError: easyocr`                         | Faltan dependencias                | `pip install -r requirements.txt`                   |
| EasyOCR descarga lento                         | Primera ejecución                  | Espera; los modelos se cachean en `~/.EasyOCR/`     |
| `OOM` (out-of-memory) al entrenar              | Batch demasiado grande             | Reduce `--batch-size 32` o `--n-muestras 10000`     |
| No detecta texto en escenas con poca luz       | Imagen muy oscura                  | Aplica el pre-procesado de la app antes de inferir  |
| Modelo propio predice "cadenas raras"          | Pocas épocas                       | Entrena 20+ épocas y con más muestras               |

---

## 8. Estructura final del entregable

```
entregable_ocr/
├── ocr_proyecto_ia/      <- código + notebook + app
├── docs/
│   ├── Reporte_OCR.pdf
│   └── Presentacion_OCR.pptx
└── README.md
```
