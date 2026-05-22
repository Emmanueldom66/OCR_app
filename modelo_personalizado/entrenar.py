"""
Entrenamiento del modelo CNN+CTC sobre EMNIST.

Para fines didácticos se construyen "pseudo-palabras" concatenando caracteres
EMNIST aleatoriamente, lo que permite practicar el reconocimiento de
secuencias con CTC sin necesidad de descargar IAM o Synth90k (que pesan
varios GB).

Uso:
    python entrenar.py --epochs 10 --batch-size 64 --longitud-max 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from modelo import (
    ALFABETO,
    TAM_VOCAB,
    construir_modelo_entrenamiento,
)


# ------------------------------------------------------------------
# Carga de EMNIST (subset 'byclass' = 62 clases alfanuméricas)
# ------------------------------------------------------------------
def cargar_emnist():
    """Carga EMNIST byclass desde tensorflow_datasets, o cae a MNIST."""
    try:
        import tensorflow_datasets as tfds

        ds = tfds.load("emnist/byclass", split="train", as_supervised=True)
        imgs, labels = [], []
        for img, lbl in ds.take(50_000):  # 50k para no saturar memoria
            imgs.append(img.numpy().squeeze())
            labels.append(int(lbl.numpy()))
        return np.array(imgs), np.array(labels)
    except Exception as e:
        print(f"⚠️  No se pudo cargar EMNIST ({e}). Se usa MNIST como fallback.")
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        return x_train, y_train


# ------------------------------------------------------------------
# Generador de pseudo-palabras
# ------------------------------------------------------------------
def generar_pseudo_palabras(
    caracteres: np.ndarray,
    etiquetas: np.ndarray,
    n_muestras: int,
    long_min: int = 3,
    long_max: int = 8,
    alto: int = 32,
    ancho: int = 128,
):
    """Crea imágenes 32x128 concatenando dígitos/letras EMNIST."""
    rng = np.random.default_rng(seed=42)
    X = np.zeros((n_muestras, alto, ancho), dtype="float32")
    Y = np.full((n_muestras, long_max), -1, dtype="int32")

    car_alto = caracteres.shape[1]  # 28
    car_ancho = caracteres.shape[2]

    # Escalamos los caracteres de 28x28 a 32x24 para encajar varios horizontalmente
    target_ancho = ancho // long_max  # 16

    for i in range(n_muestras):
        L = rng.integers(long_min, long_max + 1)
        indices = rng.choice(len(caracteres), size=L, replace=True)
        x_offset = 0
        for j, idx in enumerate(indices):
            car = caracteres[idx]
            car_rs = tf.image.resize(car[..., np.newaxis], (alto, target_ancho)).numpy().squeeze()
            X[i, :, x_offset : x_offset + target_ancho] = car_rs / 255.0
            Y[i, j] = etiquetas[idx]
            x_offset += target_ancho

    return X[..., np.newaxis], Y


# ------------------------------------------------------------------
# Generador de batches
# ------------------------------------------------------------------
def generador_batches(X, Y, batch_size: int):
    n = len(X)
    while True:
        idx = np.random.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            sel = idx[i : i + batch_size]
            yield {"imagen": X[sel], "etiqueta": Y[sel].astype("float32")}, np.zeros(batch_size)


# ------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-muestras", type=int, default=20_000)
    parser.add_argument("--longitud-max", type=int, default=8)
    args = parser.parse_args()

    print("📥 Cargando dataset...")
    caracteres, etiquetas = cargar_emnist()
    print(f"   {len(caracteres)} caracteres disponibles")

    print("✂️  Generando pseudo-palabras...")
    X, Y = generar_pseudo_palabras(
        caracteres,
        etiquetas,
        n_muestras=args.n_muestras,
        long_max=args.longitud_max,
    )
    print(f"   X: {X.shape}, Y: {Y.shape}")

    print("🧠 Construyendo modelo CNN+CTC...")
    modelo = construir_modelo_entrenamiento()
    modelo.summary()

    salida_pesos = Path(__file__).parent / "pesos.h5"
    callbacks = [
        ModelCheckpoint(str(salida_pesos), save_best_only=False, save_weights_only=True),
        ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ]

    print("🚀 Entrenando...")
    steps = len(X) // args.batch_size
    modelo.fit(
        generador_batches(X, Y, args.batch_size),
        steps_per_epoch=steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"✅ Pesos guardados en {salida_pesos}")


if __name__ == "__main__":
    main()
