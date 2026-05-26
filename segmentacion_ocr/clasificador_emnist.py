# segmentacion_ocr/clasificador_emnist.py
"""
Clasificador CNN para caracteres EMNIST.
Se encarga de entrenar un modelo que reconoce caracteres individuales.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TAM_VOCAB = len(ALFABETO)  # 62 clases

# ------------------------------------------------------------------
# Construcción del clasificador CNN
# ------------------------------------------------------------------
def construir_clasificador(input_shape=(28, 28, 1)):
    """Modelo CNN simple para clasificación de caracteres aislados."""
    
    inputs = layers.Input(shape=input_shape, name="imagen")
    
    # Bloque convolucional 1
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 14x14
    
    # Bloque convolucional 2
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 7x7
    
    # Bloque convolucional 3
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 3x3
    
    # Clasificación
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(TAM_VOCAB, activation="softmax", name="salida")(x)
    
    modelo = models.Model(inputs=inputs, outputs=outputs, name="CNN_Clasificador")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


# ------------------------------------------------------------------
# Carga del dataset EMNIST
# ------------------------------------------------------------------
def cargar_emnist_byclass(num_muestras: int = 100000):
    """Carga EMNIST ByClass y devuelve imágenes y etiquetas."""
    try:
        import tensorflow_datasets as tfds
        
        # Cargar EMNIST ByClass (62 clases alfanuméricas)
        ds = tfds.load("emnist/byclass", split="train", as_supervised=True)
        
        imgs, labels = [], []
        for i, (img, lbl) in enumerate(ds):
            if i >= num_muestras:
                break
            # Convertir imagen (28x28) a formato correcto
            img_np = img.numpy().squeeze()  # 28x28
            # EMNIST viene rotada 90 grados y volteada, corregir
            img_np = np.rot90(img_np, k=1)
            img_np = np.fliplr(img_np)
            imgs.append(img_np)
            labels.append(int(lbl.numpy()))
        
        # Normalizar
        X = np.array(imgs).reshape(-1, 28, 28, 1).astype("float32") / 255.0
        Y = np.array(labels)
        
        return X, Y
    except Exception as e:
        print(f"⚠️  No se pudo cargar EMNIST ({e}). Se usará MNIST como fallback.")
        (X_train, Y_train), _ = tf.keras.datasets.mnist.load_data()
        X = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        Y = Y_train
        return X, Y


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------
def entrenar_clasificador(epochs: int = 15, batch_size: int = 64, guardar_pesos: str = None):
    """Entrena el clasificador CNN sobre EMNIST."""
    
    print("📥 Cargando EMNIST ByClass...")
    X, Y = cargar_emnist_byclass(num_muestras=100000)
    print(f"   X: {X.shape}, Y: {Y.shape}")
    
    # Dividir en entrenamiento y validación
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    print("🧠 Construyendo clasificador CNN...")
    modelo = construir_clasificador()
    modelo.summary()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]
    
    # Guardar pesos
    if guardar_pesos is None:
        guardar_pesos = "clasificador_pesos.weights.h5"
    
    checkpoint = callbacks.ModelCheckpoint(
        guardar_pesos, save_best_only=True, save_weights_only=True, verbose=1
    )
    callbacks_list.append(checkpoint)
    
    print("🚀 Entrenando...")
    historial = modelo.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    print(f"✅ Clasificador guardado en {guardar_pesos}")
    return modelo, historial


# ------------------------------------------------------------------
# Visualización de predicciones
# ------------------------------------------------------------------
def predecir_imagen(modelo, img_28x28: np.ndarray) -> str:
    """Predice el carácter a partir de una imagen 28x28."""
    if img_28x28.ndim == 2:
        img_28x28 = img_28x28.reshape(1, 28, 28, 1)
    elif img_28x28.ndim == 3 and img_28x28.shape[2] != 1:
        # Convertir a escala de grises si es necesario
        img_28x28 = img_28x28.mean(axis=2).reshape(1, 28, 28, 1)
    
    if img_28x28.max() > 1.0:
        img_28x28 = img_28x28 / 255.0
    
    probs = modelo.predict(img_28x28, verbose=0)
    idx = np.argmax(probs[0])
    return ALFABETO[idx], probs[0]


def mostrar_ejemplos(modelo, X, Y, num_ejemplos=16):
    """Muestra ejemplos de predicción del clasificador."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, img, lbl_real in zip(axes.flat, X[:num_ejemplos], Y[:num_ejemplos]):
        pred_char, probs = predecir_imagen(modelo, img)
        real_char = ALFABETO[lbl_real] if lbl_real < TAM_VOCAB else "?"
        conf = np.max(probs) * 100
        
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Real: {real_char}\nPred: {pred_char} ({conf:.1f}%)")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--guardar-pesos", type=str, default="clasificador_pesos.weights.h5")
    args = parser.parse_args()
    
    modelo, _ = entrenar_clasificador(args.epochs, args.batch_size, args.guardar_pesos)