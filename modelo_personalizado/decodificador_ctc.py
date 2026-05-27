"""
Decodificadores de CTC: greedy y beam search.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from modelo import ALFABETO


def _indices_greedy(predicciones: np.ndarray) -> np.ndarray:
    if predicciones.ndim == 3:
        predicciones = predicciones[0]
    return np.argmax(predicciones, axis=-1)


def decodificar_greedy(predicciones: np.ndarray) -> str:
    """Decodificación greedy: argmax + colapso de repeticiones + eliminación del blanco."""
    texto, _ = decodificar_greedy_con_confianza(predicciones)
    return texto


def decodificar_greedy_con_confianza(predicciones: np.ndarray) -> tuple[str, float]:
    """Greedy + confianza media de los caracteres emitidos (prob. softmax)."""
    if predicciones.ndim == 3:
        predicciones = predicciones[0]
    indices = _indices_greedy(predicciones)

    salida: list[str] = []
    confidencias: list[float] = []
    anterior = -1
    for t, idx in enumerate(indices):
        if idx == anterior or idx == len(ALFABETO):
            anterior = idx
            continue
        salida.append(ALFABETO[idx])
        confidencias.append(float(predicciones[t, idx]))
        anterior = idx
    confianza = float(np.mean(confidencias)) if confidencias else 0.0
    return "".join(salida), confianza


def decodificar_beam(predicciones: np.ndarray, ancho_beam: int = 10) -> str:
    """Decodificación beam search usando la implementación de TensorFlow."""
    if predicciones.ndim == 2:
        predicciones = predicciones[np.newaxis, ...]
    # TF espera (T, B, V); aquí transponemos
    log_probs = np.log(predicciones.transpose(1, 0, 2) + 1e-9)
    long_secuencia = np.array([predicciones.shape[1]])

    decoded, _ = tf.nn.ctc_beam_search_decoder(
        inputs=log_probs.astype("float32"),
        sequence_length=long_secuencia,
        beam_width=ancho_beam,
        top_paths=1,
    )
    indices = tf.sparse.to_dense(decoded[0]).numpy()[0]
    return "".join(ALFABETO[i] for i in indices if 0 <= i < len(ALFABETO))
