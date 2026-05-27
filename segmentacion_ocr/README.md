# Enfoque OCR: Segmentación por contornos + Clasificador CNN

Este módulo implementa un pipeline OCR alternativo al modelo CRNN+CTC. En lugar de tratar la palabra completa como una secuencia, **segmenta cada carácter individual** y lo clasifica con una red neuronal convolutional (CNN) entrenada sobre EMNIST.

---

## ¿Cómo funciona?

El pipeline se compone de cuatro etapas claramente diferenciadas:

1. **Preprocesamiento**  
   - Conversión a escala de grises.  
   - Binarización con el método de Otsu (fondo blanco, texto negro).  
   - (Opcional) Corrección de inclinación (*deskew*).

2. **Segmentación de caracteres**  
   - Se detectan los contornos de los objetos blancos (los caracteres) en la imagen binarizada.  
   - Se filtran contornos demasiado pequeños (ruido).  
   - Se ordenan los contornos de izquierda a derecha.  
   - Cada carácter se recorta y se redimensiona a 28×28 píxeles (tamaño de entrada del clasificador).

3. **Clasificación por carácter**  
   - Se utiliza una CNN entrenada sobre **EMNIST ByClass** (62 clases: dígitos, letras mayúsculas y minúsculas).  
   - La CNN tiene tres bloques convolucionales + dropout y una salida softmax.  
   - Para cada carácter recortado, la CNN devuelve la clase más probable y su confianza.

4. **Ensamblaje**  
   - Los caracteres clasificados se concatenan en el orden de izquierda a derecha para formar la palabra final.

---

## Arquitectura del clasificador CNN

```
Input: 28×28×1
↓
Conv2D 32 (3×3) + BN + ReLU + MaxPool (2×2)
↓
Conv2D 64 (3×3) + BN + ReLU + MaxPool (2×2)
↓
Conv2D 128 (3×3) + BN + ReLU + MaxPool (2×2)
↓
Flatten + Dropout(0.5)
↓
Dense 256 + ReLU + Dropout(0.3)
↓
Dense 62 + Softmax
```

- **Parámetros totales**: ~404 670  
- **Precisión en validación**: > 99% sobre EMNIST después de 15 épocas.

---

## Entrenamiento del clasificador

### Requisitos adicionales
- `tensorflow-datasets` (para cargar EMNIST)  
- `importlib-resources` (solo si `tfds` falla)

Puedes instalarlos con:
```bash
pip install tensorflow-datasets importlib-resources
```

### Comando de entrenamiento
```bash
cd segmentacion_ocr
python clasificador_emnist.py --epochs 20 --batch-size 64
```

El script:
- Descarga EMNIST ByClass (si no está en `~/tensorflow_datasets`).
- Corrige la orientación de las imágenes (rotación y volteo).
- Entrena la CNN y guarda los mejores pesos en `clasificador_pesos.weights.h5`.
- Muestra métricas de precisión y pérdida.

---

## Uso del pipeline completo

### Inferencia sobre una imagen (línea de comandos)

```bash
python pipeline_segmentacion.py --imagen ruta/a/imagen.png --pesos clasificador_pesos.weights.h5
```

### Desde Python

```python
from segmentacion_ocr.pipeline_segmentacion import cargar_clasificador, reconocer_palabra_segmentacion

modelo = cargar_clasificador("ruta/a/pesos.weights.h5")
texto, detalles = reconocer_palabra_segmentacion(modelo, imagen_rgb)
print("Palabra:", texto)
for i, (car, conf) in enumerate(detalles):
    print(f"  Carácter {i+1}: {car} (confianza {conf:.2%})")
```

### Integración con la app Streamlit

En la barra lateral de `app/streamlit_app.py` marca la casilla **"Probar también el enfoque de Segmentación + CNN"**. La aplicación cargará automáticamente el clasificador y mostrará:

- La palabra reconocida.
- Confianza por carácter.
- Pasos intermedios: imagen binarizada, contornos detectados, caracteres extraídos.

---

## Limitaciones conocidas

- **Fondo limpio**: funciona mejor sobre fondos claros y texto con contraste alto.  
- **Separación de caracteres**: requiere que los caracteres no se toquen entre sí (ideal para texto impreso o manuscrito bien espaciado).  
- **Inclinación moderada**: la corrección automática de inclinación ayuda, pero no funciona con rotaciones extremas (> 15°).  
- **Longitud de palabra**: no tiene límite teórico (a diferencia del modelo CRNN+CTC que tiene un paso temporal fijo).

---

## Comparación con el modelo CRNN+CTC

| Característica              | Segmentación + CNN                  | CRNN+CTC (modelo propio)           |
|-----------------------------|--------------------------------------|--------------------------------------|
| **Entrenamiento**           | Sobre caracteres aislados (EMNIST)   | Sobre secuencias completas (pseudo-palabras) |
| **Longitud de palabra**     | Ilimitada                            | Fija (hasta 8 caracteres)            |
| **Interpretabilidad**       | Alta (cada carácter visible)         | Baja                                 |
| **Tamaño del modelo**       | ~1.5 MB (clasificador)               | ~17 MB                               |
| **Velocidad inferencia**    | ~50 ms por palabra (CPU)             | ~80 ms                               |
| **Requiere alineamiento**   | No                                   | Sí (CTC)                             |
| **Robustez a ruido**        | Baja (depende de la segmentación)    | Media                                |

---

## Referencias

- EMNIST dataset: Cohen et al. (2017)  
- Método de segmentación por contornos: técnica clásica de OpenCV.  
- Arquitectura CNN basada en LeNet-5 adaptada a 62 clases.