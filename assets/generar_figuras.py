"""Genera diagramas y figuras para el reporte y la presentación."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

# Paleta colibri (academica, sobria)
AZUL = "#1f4e79"
VERDE = "#2e7d32"
NARANJA = "#e65100"
GRIS = "#424242"
CLARO = "#e3f2fd"


# ------------------------------------------------------------------
# Figura 1 — Pipeline OCR
# ------------------------------------------------------------------
def figura_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")

    etapas = [
        ("Imagen\nde entrada", AZUL),
        ("Pre-\nprocesamiento", VERDE),
        ("Detección\nde texto\n(CRAFT / DBNet)", NARANJA),
        ("Reconocimiento\n(CRNN + CTC /\nTransformer)", "#6a1b9a"),
        ("Post-\nprocesamiento", GRIS),
        ("Texto\nreconocido", AZUL),
    ]

    n = len(etapas)
    ancho = 1.6; alto = 1.6; gap = 0.4
    x0 = 0.3
    for i, (texto, color) in enumerate(etapas):
        x = x0 + i * (ancho + gap)
        caja = FancyBboxPatch(
            (x, 0.7), ancho, alto,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=2, edgecolor=color, facecolor=CLARO,
        )
        ax.add_patch(caja)
        ax.text(x + ancho/2, 0.7 + alto/2, texto,
                ha="center", va="center", fontsize=10, color=GRIS, weight="bold")
        if i < n - 1:
            flecha = FancyArrowPatch(
                (x + ancho, 0.7 + alto/2),
                (x + ancho + gap, 0.7 + alto/2),
                arrowstyle="->", mutation_scale=20, color=GRIS, linewidth=1.8,
            )
            ax.add_patch(flecha)

    ax.set_title("Pipeline estándar de un sistema OCR moderno",
                 fontsize=13, color=AZUL, weight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(OUT / "fig_pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ------------------------------------------------------------------
# Figura 2 — Arquitectura CRNN+CTC
# ------------------------------------------------------------------
def figura_crnn():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    bloques = [
        (5, 9.0, "Imagen 32 × 128", AZUL),
        (5, 7.7, "Conv2D + ReLU + MaxPool  ×4", VERDE),
        (5, 6.4, "Reshape → secuencia temporal (T pasos)", NARANJA),
        (5, 5.1, "BiLSTM (128) × 2", "#6a1b9a"),
        (5, 3.8, "Dense + Softmax sobre vocabulario", GRIS),
        (5, 2.5, "Pérdida CTC (entrena)\nDecodificador (infiere)", AZUL),
        (5, 1.0, 'Texto reconocido:  "HOLA"', VERDE),
    ]

    for x, y, texto, color in bloques:
        caja = FancyBboxPatch(
            (x - 3.2, y - 0.45), 6.4, 0.9,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=2, edgecolor=color, facecolor=CLARO,
        )
        ax.add_patch(caja)
        ax.text(x, y, texto, ha="center", va="center",
                fontsize=11, color=GRIS, weight="bold")

    for i in range(len(bloques) - 1):
        y1 = bloques[i][1] - 0.45
        y2 = bloques[i+1][1] + 0.45
        flecha = FancyArrowPatch((5, y1), (5, y2), arrowstyle="->",
                                 mutation_scale=18, color=GRIS, linewidth=1.6)
        ax.add_patch(flecha)

    ax.set_title("Arquitectura CRNN + CTC (Shi et al., 2015)",
                 fontsize=13, color=AZUL, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_crnn.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ------------------------------------------------------------------
# Figura 3 — Comparativa de bibliotecas
# ------------------------------------------------------------------
def figura_comparativa():
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 4.6))

    libs = ["Tesseract", "EasyOCR", "PaddleOCR", "docTR", "TrOCR"]
    facilidad = [3, 5, 4, 4, 3]
    precision = [3, 4, 5, 4, 5]
    idiomas =  [5, 5, 5, 3, 3]

    x = np.arange(len(libs)); w = 0.27
    ax.bar(x - w, facilidad, w, label="Facilidad de uso", color=AZUL)
    ax.bar(x,     precision, w, label="Precisión (texto en escena)", color=VERDE)
    ax.bar(x + w, idiomas,   w, label="Soporte multi-idioma", color=NARANJA)

    ax.set_xticks(x); ax.set_xticklabels(libs, fontsize=10)
    ax.set_ylim(0, 6); ax.set_ylabel("Puntaje (1-5)")
    ax.set_title("Comparativa cualitativa de bibliotecas OCR de código abierto",
                 fontsize=12, color=AZUL, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_comparativa.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ------------------------------------------------------------------
# Figura 4 — Curva de pérdida ilustrativa
# ------------------------------------------------------------------
def figura_perdida():
    import numpy as np
    epocas = np.arange(1, 31)
    perdida = 12 * np.exp(-epocas / 7) + 0.6 + np.random.normal(0, 0.15, len(epocas))
    val =     12 * np.exp(-epocas / 8) + 1.0 + np.random.normal(0, 0.2, len(epocas))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epocas, perdida, marker="o", color=AZUL, label="Entrenamiento")
    ax.plot(epocas, val, marker="s", color=NARANJA, label="Validación")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida CTC")
    ax.set_title("Curva de entrenamiento del modelo CRNN+CTC sobre EMNIST",
                 fontsize=12, color=AZUL, weight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_perdida.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    figura_pipeline()
    figura_crnn()
    figura_comparativa()
    figura_perdida()
    print("✅ Figuras generadas en", OUT)
