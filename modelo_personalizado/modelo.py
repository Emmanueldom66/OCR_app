"""
Arquitectura CNN+CTC para reconocimiento de secuencias de caracteres.

Basado en la red CRNN de Shi et al. (2015):
"An End-to-End Trainable Neural Network for Image-based Sequence Recognition"
https://arxiv.org/abs/1507.05717

Entrada:  imagen en escala de grises de 32 x 128 pixeles
Salida:   secuencia de probabilidades sobre el vocabulario (CTC)
"""

from __future__ import annotations

import tensorflow as tf

layers = tf.keras.layers
Model = tf.keras.Model


# Vocabulario: dígitos + letras mayúsculas (compatible con EMNIST 'byclass')
ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TAM_VOCAB = len(ALFABETO) + 1  # +1 para el token blanco del CTC


# ------------------------------------------------------------------
# Capa CTC (calcula la pérdida durante entrenamiento)
# ------------------------------------------------------------------
class CapaCTC(layers.Layer):
    """Capa que añade la pérdida CTC al modelo durante entrenamiento."""

    def __init__(self, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        batch = tf.shape(y_pred)[0]
        y_true = tf.cast(y_true, tf.int32)
        # Etiquetas con -1 de relleno: longitud real y valores >= 0 para CTC
        mascara = tf.not_equal(y_true, -1)
        long_etiqueta = tf.reduce_sum(tf.cast(mascara, tf.int32), axis=1, keepdims=True)
        y_true = tf.where(mascara, y_true, tf.zeros_like(y_true))

        long_entrada = tf.fill([batch, 1], tf.shape(y_pred)[1])

        perdida = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, long_entrada, long_etiqueta
        )
        self.add_loss(tf.reduce_mean(perdida))
        # Salida ficticia: evita ciclo en el grafo (Keras 3) y coincide con y=np.zeros del fit
        return tf.zeros((batch, 1), dtype=y_pred.dtype)


# ------------------------------------------------------------------
# Bloque CNN
# ------------------------------------------------------------------
def _bloque_cnn(x):
    """Extractor de características convolucional."""
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16x64

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8x32

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # 4x32

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # 2x32

    x = layers.Conv2D(256, (2, 2), activation="relu")(x)  # 1x31
    return x


# ------------------------------------------------------------------
# Modelo de entrenamiento (con etiquetas + CTC)
# ------------------------------------------------------------------
def construir_modelo_entrenamiento(altura: int = 32, ancho: int = 128) -> Model:
    """Construye el modelo CRNN+CTC para entrenar."""
    entrada_img = layers.Input(shape=(altura, ancho, 1), name="imagen")
    etiqueta = layers.Input(shape=(None,), dtype="int32", name="etiqueta")

    # CNN
    x = _bloque_cnn(entrada_img)
    # (batch, 1, T, C) -> (batch, T, C)
    x = layers.Reshape(target_shape=(-1, x.shape[-1]))(x)

    # RNN bidireccional
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    # Capa densa de salida (softmax sobre vocabulario)
    y_pred = layers.Dense(TAM_VOCAB, activation="softmax", name="salida_softmax")(x)

    # Pérdida CTC
    salida = CapaCTC(name="ctc")(etiqueta, y_pred)

    modelo = Model(inputs=[entrada_img, etiqueta], outputs=salida, name="CRNN_CTC")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return modelo


# ------------------------------------------------------------------
# Modelo de inferencia (sin la cabeza CTC, sólo softmax)
# ------------------------------------------------------------------
def construir_modelo_inferencia(altura: int = 32, ancho: int = 128) -> Model:
    """Versión sin CTC para predecir nuevas imágenes."""
    entrada_img = layers.Input(shape=(altura, ancho, 1), name="imagen")
    x = _bloque_cnn(entrada_img)
    x = layers.Reshape(target_shape=(-1, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    y_pred = layers.Dense(TAM_VOCAB, activation="softmax", name="salida_softmax")(x)
    return Model(inputs=entrada_img, outputs=y_pred, name="CRNN_CTC_inferencia")
