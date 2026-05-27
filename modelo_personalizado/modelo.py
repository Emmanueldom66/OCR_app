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
from tensorflow.keras import layers, Model


# Vocabulario: dígitos + letras mayúsculas (compatible con EMNIST 'byclass')
ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TAM_VOCAB = len(ALFABETO) + 1  # +1 para el token blanco del CTC


# ------------------------------------------------------------------
# Capa CTC (calcula la pérdida durante entrenamiento)
# ------------------------------------------------------------------
"""
class CapaCTC(layers.Layer):
    #Capa que añade la pérdida CTC al modelo durante entrenamiento.

    def __init__(self, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        batch = tf.shape(y_pred)[0]
        long_entrada = tf.shape(y_pred)[1]
        long_etiqueta = tf.shape(y_true)[1]

        long_entrada = long_entrada * tf.ones((batch, 1), dtype="int64")
        long_etiqueta = long_etiqueta * tf.ones((batch, 1), dtype="int64")

        perdida = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, long_entrada, long_etiqueta
        )
        self.add_loss(tf.reduce_mean(perdida))
        return y_pred
"""

class CapaCTC(layers.Layer):
    """Capa CTC que añade la pérdida correctamente."""
    
    def __init__(self, blank_index=62, **kwargs):
        super().__init__(**kwargs)
        self.blank_index = blank_index  # índice del token blanco (último del vocabulario)

    def call(self, y_true, y_pred):
        # y_true: shape (batch, max_label_len), con -1 para padding
        # y_pred: shape (batch, time, num_classes)
        batch = tf.shape(y_pred)[0]
        time = tf.shape(y_pred)[1]

        # Logits (si y_pred ya es softmax, tomamos log)
        logits = tf.math.log(y_pred + 1e-9)

        # Longitud de la entrada (todas las secuencias tienen la misma longitud 'time')
        input_length = tf.fill([batch], time)

        # Longitudes reales de las etiquetas (contando valores != -1)
        mask = tf.cast(tf.not_equal(y_true, -1), tf.int32)
        label_length = tf.reduce_sum(mask, axis=1)

        # y_true debe ser int32 para tf.nn.ctc_loss
        y_true_int = tf.cast(y_true, tf.int32)

        # Calcular pérdida CTC
        loss = tf.nn.ctc_loss(
            labels=y_true_int,
            logits=logits,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,          # porque nuestro logits es (batch, time, classes)
            blank_index=self.blank_index
        )
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

    def compute_output_shape(self, input_shape):
        # input_shape es (shape_y_true, shape_y_pred)
        # La capa devuelve y_pred, así que retornamos shape_y_pred
        return input_shape[1]

    def compute_output_spec(self, inputs, training=None):
        return inputs[1]


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
    etiqueta = layers.Input(shape=(None,), dtype="float32", name="etiqueta")

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
    #salida = CapaCTC(name="ctc")(etiqueta, y_pred)
    # Dentro de construir_modelo_entrenamiento
    salida = CapaCTC(blank_index=TAM_VOCAB - 1, name="ctc")(etiqueta, y_pred)

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
