"""
Inferencia con el modelo CNN+CTC entrenado.

Uso:
    python inferencia.py --imagen ruta/a/imagen.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from decodificador_ctc import decodificar_greedy, decodificar_beam
from modelo import construir_modelo_inferencia


def cargar_y_normalizar(ruta: str, alto: int = 32, ancho: int = 128) -> np.ndarray:
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ruta)
    img = cv2.resize(img, (ancho, alto))
    img = img.astype("float32") / 255.0
    return img[np.newaxis, ..., np.newaxis]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", required=True)
    parser.add_argument("--pesos", default=str(Path(__file__).parent / "pesos.h5"))
    parser.add_argument("--metodo", choices=["greedy", "beam"], default="greedy")
    args = parser.parse_args()

    print(f"📥 Cargando modelo desde {args.pesos}")
    modelo = construir_modelo_inferencia()
    modelo.load_weights(args.pesos)

    entrada = cargar_y_normalizar(args.imagen)
    prediccion = modelo.predict(entrada, verbose=0)

    if args.metodo == "greedy":
        texto = decodificar_greedy(prediccion)
    else:
        texto = decodificar_beam(prediccion)

    print(f"📝 Texto reconocido: '{texto}'")


if __name__ == "__main__":
    main()
