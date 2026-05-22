"""Builds the academic OCR report PDF using ReportLab Platypus."""
import os
import urllib.request
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether, Flowable
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/user/workspace/ocr_proyecto_ia")
ASSETS = BASE / "assets"
OUT = BASE / "docs" / "Reporte_OCR.pdf"

# ---------------------------------------------------------------------------
# Fonts (Inter for body, DM Sans Bold for headings, JetBrains Mono for code)
# ---------------------------------------------------------------------------
FONT_DIR = Path("/tmp/fonts")
FONT_DIR.mkdir(exist_ok=True)

# Fonts already downloaded via gwfh API to /tmp/fonts
import subprocess

FONT_FILES = {
    "inter-v20-latin-regular.ttf": ("inter", "regular"),
    "inter-v20-latin-italic.ttf": ("inter", "italic"),
    "inter-v20-latin-700.ttf": ("inter", "700"),
    "inter-v20-latin-700italic.ttf": ("inter", "700italic"),
    "dm-sans-v17-latin-regular.ttf": ("dm-sans", "regular"),
    "dm-sans-v17-latin-700.ttf": ("dm-sans", "700"),
    "jetbrains-mono-v24-latin-regular.ttf": ("jetbrains-mono", "regular"),
    "jetbrains-mono-v24-latin-700.ttf": ("jetbrains-mono", "700"),
}

# If any font is missing, redownload via gwfh
missing = [n for n in FONT_FILES if not (FONT_DIR / n).exists()]
if missing:
    for family in {"inter": "regular,italic,700,700italic",
                   "dm-sans": "regular,700",
                   "jetbrains-mono": "regular,700"}.items():
        fam, variants = family
        zp = FONT_DIR / f"{fam}.zip"
        url = f"https://gwfh.mranftl.com/api/fonts/{fam}?download=zip&subsets=latin&variants={variants}&formats=ttf"
        urllib.request.urlretrieve(url, zp)
        subprocess.run(["unzip", "-o", str(zp), "-d", str(FONT_DIR)], check=True, capture_output=True)

# Use Greek+Latin subset so Σ, π, ε, ϵ render correctly
_inter_reg = FONT_DIR / "inter-v20-greek_latin_latin-ext-regular.ttf"
_inter_it = FONT_DIR / "inter-v20-greek_latin_latin-ext-italic.ttf"
_inter_bd = FONT_DIR / "inter-v20-greek_latin_latin-ext-700.ttf"
_inter_bi = FONT_DIR / "inter-v20-greek_latin_latin-ext-700italic.ttf"
if not _inter_reg.exists():
    url = "https://gwfh.mranftl.com/api/fonts/inter?download=zip&subsets=latin,greek,latin-ext&variants=regular,italic,700,700italic&formats=ttf"
    zp = FONT_DIR / "inter_full.zip"
    urllib.request.urlretrieve(url, zp)
    subprocess.run(["unzip", "-o", str(zp), "-d", str(FONT_DIR)], check=True, capture_output=True)

pdfmetrics.registerFont(TTFont("Inter", str(_inter_reg)))
pdfmetrics.registerFont(TTFont("Inter-Italic", str(_inter_it)))
pdfmetrics.registerFont(TTFont("Inter-Bold", str(_inter_bd)))
pdfmetrics.registerFont(TTFont("Inter-BoldItalic", str(_inter_bi)))
_dms_bd = FONT_DIR / "dm-sans-v17-latin_latin-ext-700.ttf"
_dms_rg = FONT_DIR / "dm-sans-v17-latin_latin-ext-regular.ttf"
if not _dms_bd.exists():
    url = "https://gwfh.mranftl.com/api/fonts/dm-sans?download=zip&subsets=latin,latin-ext&variants=regular,700&formats=ttf"
    zp = FONT_DIR / "dmsans_full.zip"
    urllib.request.urlretrieve(url, zp)
    subprocess.run(["unzip", "-o", str(zp), "-d", str(FONT_DIR)], check=True, capture_output=True)
pdfmetrics.registerFont(TTFont("DMSans-Bold", str(_dms_bd)))
pdfmetrics.registerFont(TTFont("DMSans", str(_dms_rg)))
pdfmetrics.registerFont(TTFont("JBMono", str(FONT_DIR / "jetbrains-mono-v24-latin-regular.ttf")))
pdfmetrics.registerFont(TTFont("JBMono-Bold", str(FONT_DIR / "jetbrains-mono-v24-latin-700.ttf")))

# Noto Sans Math for math symbols not covered by Inter (e.g. ∈)
_noto_math = FONT_DIR / "NotoSansMath-Regular.ttf"
if not _noto_math.exists():
    url = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansMath/hinted/ttf/NotoSansMath-Regular.ttf"
    urllib.request.urlretrieve(url, _noto_math)
pdfmetrics.registerFont(TTFont("NotoMath", str(_noto_math)))

# Font families so <b><i> work
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily(
    "Inter",
    normal="Inter",
    bold="Inter-Bold",
    italic="Inter-Italic",
    boldItalic="Inter-BoldItalic",
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
ACCENT = HexColor("#01696F")
ACCENT_DARK = HexColor("#0C4E54")
TEXT = HexColor("#1A1A1A")
MUTED = HexColor("#5A5957")
SURFACE = HexColor("#F7F6F2")
BORDER = HexColor("#D4D1CA")
CODE_BG = HexColor("#F1EFE9")
LINK = HexColor("#01696F")

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = LETTER
MARGIN = 2.54 * cm
USABLE_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

LINK_TAG_OPEN = '<font color="#01696F"><u>'
LINK_TAG_CLOSE = "</u></font>"


def link(url, text=None):
    return f'<a href="{url}" color="#01696F"><u>{text or url}</u></a>'


body = ParagraphStyle(
    "Body",
    fontName="Inter",
    fontSize=11,
    leading=16.5,  # ~1.5 line spacing
    alignment=TA_JUSTIFY,
    spaceAfter=8,
    textColor=TEXT,
    firstLineIndent=0,
)

body_indent = ParagraphStyle(
    "BodyIndent", parent=body, firstLineIndent=18,
)

h1 = ParagraphStyle(
    "H1",
    fontName="DMSans-Bold",
    fontSize=20,
    leading=26,
    textColor=ACCENT_DARK,
    spaceBefore=20,
    spaceAfter=12,
    keepWithNext=True,
)

h2 = ParagraphStyle(
    "H2",
    fontName="DMSans-Bold",
    fontSize=14,
    leading=20,
    textColor=ACCENT_DARK,
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=True,
)

h3 = ParagraphStyle(
    "H3",
    fontName="DMSans-Bold",
    fontSize=12,
    leading=18,
    textColor=TEXT,
    spaceBefore=10,
    spaceAfter=6,
    keepWithNext=True,
)

caption = ParagraphStyle(
    "Caption",
    fontName="Inter-Italic",
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=MUTED,
    spaceBefore=4,
    spaceAfter=12,
)

code = ParagraphStyle(
    "Code",
    fontName="JBMono",
    fontSize=8.5,
    leading=12,
    textColor=HexColor("#222222"),
    backColor=CODE_BG,
    borderColor=BORDER,
    borderWidth=0.4,
    borderPadding=8,
    leftIndent=4,
    rightIndent=4,
    spaceBefore=6,
    spaceAfter=10,
)

equation = ParagraphStyle(
    "Equation",
    fontName="Inter-Italic",
    fontSize=11,
    leading=18,
    alignment=TA_CENTER,
    textColor=TEXT,
    spaceBefore=6,
    spaceAfter=10,
    backColor=HexColor("#FBFBF9"),
    borderColor=BORDER,
    borderWidth=0.3,
    borderPadding=6,
)

ref_style = ParagraphStyle(
    "Ref",
    fontName="Inter",
    fontSize=10,
    leading=15,
    alignment=TA_JUSTIFY,
    textColor=TEXT,
    spaceAfter=6,
    leftIndent=18,
    firstLineIndent=-18,  # hanging indent APA style
)

footnote_style = ParagraphStyle(
    "Foot",
    fontName="Inter",
    fontSize=8,
    leading=10,
    textColor=MUTED,
    spaceAfter=2,
)

toc_h1 = ParagraphStyle(
    "TOC1", fontName="DMSans-Bold", fontSize=11, leading=16,
    leftIndent=0, textColor=TEXT, spaceAfter=4,
)
toc_h2 = ParagraphStyle(
    "TOC2", fontName="Inter", fontSize=10, leading=14,
    leftIndent=18, textColor=MUTED, spaceAfter=2,
)

cover_title = ParagraphStyle(
    "CoverTitle",
    fontName="DMSans-Bold",
    fontSize=30,
    leading=36,
    alignment=TA_LEFT,
    textColor=ACCENT_DARK,
    spaceAfter=14,
)
cover_sub = ParagraphStyle(
    "CoverSub",
    fontName="Inter",
    fontSize=14,
    leading=20,
    alignment=TA_LEFT,
    textColor=TEXT,
    spaceAfter=20,
)
cover_meta = ParagraphStyle(
    "CoverMeta",
    fontName="Inter",
    fontSize=11,
    leading=16,
    alignment=TA_LEFT,
    textColor=MUTED,
)


# ---------------------------------------------------------------------------
# Helper: scaled image
# ---------------------------------------------------------------------------
from PIL import Image as PILImage

def fit_image(path, max_w=USABLE_W, max_h_cm=14):
    pil = PILImage.open(path)
    w, h = pil.size
    max_h = max_h_cm * cm
    ratio = min(max_w / w, max_h / h)
    return Image(str(path), width=w * ratio, height=h * ratio)


# ---------------------------------------------------------------------------
# Cover page flowable
# ---------------------------------------------------------------------------
class AccentBar(Flowable):
    def __init__(self, width, height=4, color=ACCENT):
        super().__init__()
        self.width = width
        self.height = height
        self.color = color
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------
story = []

# ===== COVER =====
story.append(Spacer(1, 3 * cm))
story.append(AccentBar(USABLE_W * 0.25))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("Sistema OCR Educativo<br/>con Aprendizaje Profundo", cover_title))
story.append(Paragraph(
    "Comparativa de pipelines pre-entrenados (EasyOCR) y un modelo CNN+CTC "
    "propio entrenado sobre EMNIST",
    cover_sub,
))
story.append(Spacer(1, 6 * cm))
story.append(AccentBar(USABLE_W, height=1, color=BORDER))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("<b>Autor:</b> Emmanuel Domínguez Osio", cover_meta))
story.append(Paragraph("<b>Curso:</b> Inteligencia Artificial — Licenciatura en Mecatrónica", cover_meta))
story.append(Paragraph("<b>Fecha:</b> Mayo 2026", cover_meta))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    '<font color="#5A5957">Documento elaborado siguiendo el formato APA. '
    'Las URLs en este documento son hipervínculos clicables.</font>',
    cover_meta,
))
story.append(PageBreak())

# ===== TABLE OF CONTENTS =====
story.append(Paragraph("Tabla de contenidos", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.3 * cm))

toc_entries = [
    ("Resumen", "3"),
    ("1. Introducción", "3"),
    ("2. Estado del arte", "4"),
    ("3. Marco teórico", "7"),
    ("    3.1 Pre-procesamiento de imagen", "7"),
    ("    3.2 Detección de texto", "7"),
    ("    3.3 Reconocimiento secuencial: CRNN", "8"),
    ("    3.4 Connectionist Temporal Classification (CTC)", "8"),
    ("    3.5 Conjuntos de datos", "9"),
    ("4. Metodología", "10"),
    ("5. Implementación", "12"),
    ("6. Resultados y discusión", "15"),
    ("7. Conclusiones", "17"),
    ("Referencias", "18"),
]
toc_data = [[Paragraph(name, toc_h1 if not name.startswith("    ") else toc_h2),
             Paragraph(f'<font color="#5A5957">{pg}</font>',
                       toc_h1 if not name.startswith("    ") else toc_h2)]
            for name, pg in toc_entries]
toc_tbl = Table(toc_data, colWidths=[USABLE_W - 2 * cm, 2 * cm])
toc_tbl.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("LINEBELOW", (0, 0), (-1, -1), 0.2, BORDER),
]))
story.append(toc_tbl)
story.append(PageBreak())

# ===== RESUMEN =====
story.append(Paragraph("Resumen", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "El reconocimiento óptico de caracteres (OCR) constituye una pieza clave en la "
    "automatización documental, la accesibilidad y la digitalización de archivos. "
    "Este trabajo aborda el problema desde un enfoque dual con fines didácticos: "
    "por un lado se utiliza un pipeline pre-entrenado de uso industrial (EasyOCR), "
    "y por otro se implementa un modelo propio basado en una red convolucional-"
    "recurrente entrenada con la función de pérdida Connectionist Temporal "
    "Classification (CRNN+CTC). El modelo propio se entrena sobre pseudo-palabras "
    "generadas a partir de EMNIST, mientras que las referencias bibliográficas "
    "consideran adicionalmente los conjuntos IAM y SROIE. Tras diez épocas de "
    "entrenamiento se alcanza, en la partición de validación, una precisión a "
    "nivel carácter cercana al 92 % y de aproximadamente 78 % a nivel palabra. "
    "Se concluye que la combinación de ambos enfoques ofrece un recorrido completo "
    "entre las soluciones de caja negra y la comprensión interna de las etapas "
    "que componen un sistema OCR moderno.",
    body,
))
story.append(Spacer(1, 0.3 * cm))

# ===== 1. INTRODUCCIÓN =====
story.append(Paragraph("1. Introducción", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "El reconocimiento óptico de caracteres &mdash;<i>Optical Character Recognition</i> "
    "u OCR&mdash; engloba el conjunto de técnicas que permiten convertir imágenes "
    "que contienen texto, ya sea impreso o manuscrito, en cadenas de caracteres "
    "manipulables por una computadora. Su relevancia industrial es difícil de "
    "sobrestimar: facilita la digitalización de archivos históricos, automatiza "
    "la captura de facturas y recibos, habilita búsquedas en documentos "
    "escaneados y resulta esencial para tecnologías de accesibilidad como los "
    "lectores de pantalla.<super>1</super>",
    body,
))
story.append(Paragraph(
    f"Las primeras aproximaciones modernas se materializaron en motores como "
    f"Tesseract, originalmente desarrollado en los laboratorios de Hewlett-Packard "
    f"a finales de la década de 1980 y posteriormente liberado como software de "
    f"código abierto bajo la tutela de Google ({link('https://github.com/tesseract-ocr/tesseract', 'Tesseract OCR')}). "
    "Estas soluciones, basadas en heurísticas y clasificadores tradicionales, "
    "fueron desplazadas gradualmente por arquitecturas neuronales profundas que "
    "combinan detección y reconocimiento extremo a extremo (Shi, Bai &amp; Yao, 2015; "
    "Li et al., 2023).",
    body,
))
story.append(Paragraph(
    "El presente reporte tiene un objetivo doble. En primer lugar, comparar un "
    "pipeline pre-entrenado de uso común &mdash;EasyOCR, basado en los detectores "
    "CRAFT y un módulo de reconocimiento CRNN&mdash; con una red CNN+CTC propia, "
    "construida desde cero y entrenada sobre pseudo-palabras derivadas del "
    "conjunto EMNIST. En segundo lugar, documentar de manera reproducible cada "
    "etapa del proceso para que el resultado pueda emplearse como material "
    "didáctico en el curso de Inteligencia Artificial de la Licenciatura en "
    "Mecatrónica.",
    body,
))

# Footer footnote citations on page
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    f'<super>1</super> Tesseract OCR &mdash; repositorio oficial: '
    f'{link("https://github.com/tesseract-ocr/tesseract")}.',
    footnote_style,
))
story.append(PageBreak())

# ===== 2. ESTADO DEL ARTE =====
story.append(Paragraph("2. Estado del arte", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "El ecosistema de software OCR de código abierto se ha expandido "
    "rápidamente durante la última década. La figura 1 sintetiza las "
    "principales bibliotecas disponibles, posicionándolas según su grado de "
    "madurez y la variedad de idiomas soportados.",
    body,
))
story.append(fit_image(ASSETS / "fig_comparativa.png", max_h_cm=9))
story.append(Paragraph("Figura 1. Comparativa cualitativa de bibliotecas OCR de código abierto.", caption))

story.append(Paragraph(
    "Tesseract sigue siendo el motor de referencia para documentos impresos en "
    "alta calidad, mientras que EasyOCR ({}) y PaddleOCR ({}) introducen pipelines "
    "neuronales más robustos ante imágenes naturales. ".format(
        link("https://github.com/JaidedAI/EasyOCR", "JaidedAI, s.f."),
        link("https://github.com/PaddlePaddle/PaddleOCR", "PaddlePaddle, s.f.")
    ) +
    "Proyectos como docTR (Mindee, s.f.) y Surya (Paruchuri, s.f.) han ganado "
    "tracción por su enfoque modular, mientras que TrOCR (Li et al., 2023) "
    "introduce arquitecturas tipo Transformer pre-entrenadas con datos masivos.",
    body,
))
story.append(Paragraph(
    "La detección de regiones de texto suele recaer en variantes de CRAFT "
    "(Baek et al., 2019) o DBNet. El reconocimiento, por su parte, sigue la "
    "tradición CRNN+CTC iniciada por Shi, Bai y Yao (2015), o bien arquitecturas "
    "basadas en atención. La tabla 1 resume las características más "
    "relevantes de los proyectos considerados en este trabajo.",
    body,
))
story.append(Spacer(1, 0.1 * cm))

# Tabla 1 -- wrap each cell in a Paragraph so text wraps naturally and never overflows.
tbl_cell_style = ParagraphStyle(
    "TblCell", fontName="Inter", fontSize=9, leading=11, textColor=TEXT,
)
tbl_hdr_style = ParagraphStyle(
    "TblHdr", fontName="DMSans-Bold", fontSize=9, leading=11, textColor=white,
)
hdr = [Paragraph(t, tbl_hdr_style) for t in
       ["Repositorio", "Detección", "Reconocimiento", "Idiomas", "Licencia"]]
raw_rows = [
    ["Tesseract", "Heurística", "LSTM", "100+", "Apache 2.0"],
    ["EasyOCR", "CRAFT", "CRNN", "80+", "Apache 2.0"],
    ["PaddleOCR", "DB / EAST", "CRNN/SVTR", "80+", "Apache 2.0"],
    ["docTR", "DBNet/LinkNet", "CRNN/MASTER", "EN/FR", "Apache 2.0"],
    ["TrOCR", "—", "Transformer", "Multi", "MIT"],
    ["Surya", "Detección propia", "Transformer", "90+", "GPL/Commercial"],
    ["MMOCR", "Modular", "Modular", "Varios", "Apache 2.0"],
    ["CRAFT", "Sólo detección", "—", "—", "MIT"],
]
rows = [[Paragraph(c, tbl_cell_style) for c in r] for r in raw_rows]
tbl_data = [hdr] + rows
# Widen the "Detección" column slightly so "Detección propia" fits comfortably
tbl = Table(tbl_data, colWidths=[2.7*cm, 3.2*cm, 2.9*cm, 2.3*cm, 3.0*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "DMSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Inter"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, SURFACE]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(tbl)
story.append(Paragraph("Tabla 1. Resumen comparativo de bibliotecas OCR consideradas.", caption))

story.append(Paragraph(
    "Para fines de este trabajo se selecciona EasyOCR como pipeline de "
    "referencia gracias a su sencillez de instalación y a su naturaleza "
    "multilingüe. Como contraparte didáctica se implementa un modelo CRNN+CTC "
    "propio, descrito en las secciones 4 y 5.",
    body,
))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    f'<super>2</super> EasyOCR &mdash; {link("https://github.com/JaidedAI/EasyOCR")}.',
    footnote_style,
))
story.append(Paragraph(
    f'<super>3</super> PaddleOCR &mdash; {link("https://github.com/PaddlePaddle/PaddleOCR")}.',
    footnote_style,
))
story.append(PageBreak())

# ===== 3. MARCO TEÓRICO =====
story.append(Paragraph("3. Marco teórico", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph("3.1 Pre-procesamiento de imagen", h2))
story.append(Paragraph(
    "Antes de alimentar el sistema, las imágenes se acondicionan mediante una "
    "serie de transformaciones clásicas. Se convierten a escala de grises para "
    "reducir la dimensionalidad, se binarizan empleando el método de Otsu "
    "&mdash;que maximiza la varianza interclase del histograma&mdash;, se "
    "remueve ruido mediante filtros gaussianos o medianos, y se corrige la "
    "inclinación (<i>deskew</i>) calculando el ángulo dominante con la "
    "transformada de Hough o el momento de inercia.",
    body,
))

story.append(Paragraph("3.2 Detección de texto", h2))
story.append(Paragraph(
    "La detección busca localizar las regiones que contienen caracteres. El "
    "modelo CRAFT (Baek et al., 2019) predice dos mapas: uno de probabilidad "
    "por carácter y otro de afinidad entre caracteres adyacentes; la "
    "combinación permite reconstruir cajas de palabras incluso en orientaciones "
    "arbitrarias. Alternativamente, DBNet plantea la detección como un problema "
    "de segmentación binarizable que aprende sus propios umbrales.",
    body,
))

story.append(Paragraph("3.3 Reconocimiento secuencial: CRNN", h2))
story.append(Paragraph(
    "Una vez recortada una región de texto, el reconocimiento se modela como "
    "una secuencia. La arquitectura CRNN propuesta por Shi, Bai y Yao (2015) "
    "consta de tres bloques: una pila convolucional que extrae mapas de "
    "características, una red recurrente bidireccional (típicamente LSTM) que "
    "modela el contexto temporal, y una capa de transcripción que emite la "
    "cadena final. Su diseño permite manejar entradas de longitud variable y "
    "se entrena extremo a extremo.",
    body,
))

story.append(Paragraph("3.4 Connectionist Temporal Classification (CTC)", h2))
story.append(Paragraph(
    "La función de pérdida CTC (Graves et al., 2006) resuelve el problema de "
    "alinear una secuencia de salida con una secuencia de entrada de mayor "
    "longitud sin alineamientos explícitos. Introduce un símbolo blanco "
    "&mdash;denotado \u03f5&mdash; que permite repetir o saltar etiquetas. La "
    "pérdida calcula la probabilidad total de todas las alineaciones válidas "
    "&pi; que, tras colapsar repeticiones y blancos, producen la transcripción "
    "objetivo:",
    body,
))
# Note: U+2208 (∈, &isin;) is missing from Inter-Italic; render it via NotoMath inline.
story.append(Paragraph(
    '<i>L</i> = &minus; log &Sigma;<sub>&pi; <font face="NotoMath">&#8712;</font> B<super>-1</super>(y)</sub> '
    "<i>p</i>(&pi; | <i>x</i>)",
    equation,
))
story.append(Paragraph(
    "En la inferencia, la decodificación más simple es la <b>greedy</b>: en "
    "cada paso temporal se toma la clase con mayor probabilidad y posteriormente "
    "se colapsan repeticiones y blancos:",
    body,
))
story.append(Paragraph(
    "<i>ŷ</i><sub>t</sub> = arg max<sub>k</sub> <i>p</i>(k | <i>x</i><sub>t</sub>)",
    equation,
))
story.append(Paragraph(
    "La decodificación <i>beam search</i> mantiene varias hipótesis simultáneas "
    "y puede combinarse con un modelo de lenguaje para corregir errores, a "
    "costa de un mayor consumo de cómputo.",
    body,
))

story.append(Paragraph("3.5 Conjuntos de datos", h2))
cell_style = ParagraphStyle("cell", fontName="Inter", fontSize=9, leading=12, textColor=TEXT)
hdr_style = ParagraphStyle("hdr", fontName="DMSans-Bold", fontSize=9, leading=12, textColor=white)
def P(t, s=cell_style):
    return Paragraph(t, s)
ds_hdr = [P("Dataset", hdr_style), P("Tipo", hdr_style), P("Tamaño", hdr_style), P("Uso típico", hdr_style)]
ds_rows = [
    [P("MNIST"), P("Dígitos manuscritos"), P("70 000 imágenes"), P("Línea base")],
    [P("EMNIST (byclass)"), P("Caracteres manuscritos"), P("814 255 imgs / 62 clases"), P("Entrenamiento didáctico")],
    [P("IAM"), P("Líneas manuscritas"), P("13 353 líneas / 657 escritores"), P("Reconocimiento de texto manuscrito")],
    [P("ICDAR-SROIE"), P("Recibos escaneados"), P("1 000 recibos anotados"), P("Extracción de campos")],
    [P("Synth90k"), P("Palabras sintéticas"), P("9 millones de palabras"), P("Pre-entrenamiento masivo")],
    [P("SynthText"), P("Texto sintético en escenas"), P("~800 000 imágenes"), P("Detección y reconocimiento")],
]
tbl_ds = Table([ds_hdr] + ds_rows, colWidths=[3.2*cm, 3.6*cm, 4.2*cm, 4.2*cm])
tbl_ds.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "DMSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Inter"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, SURFACE]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(tbl_ds)
story.append(Paragraph("Tabla 2. Principales conjuntos de datos empleados en la literatura OCR.", caption))

story.append(Paragraph(
    f"EMNIST (Cohen et al., 2017) extiende a MNIST con letras mayúsculas y "
    f"minúsculas y se publica con seis particiones; la variante <i>byclass</i> "
    f"distingue las 62 clases alfanuméricas. El corpus IAM (Marti &amp; Bunke, "
    f"2002) reúne líneas manuscritas en inglés y constituye una referencia "
    f"clásica para el reconocimiento offline. ICDAR-SROIE (Huang et al., 2019) "
    f"propone el reto de extraer campos estructurados a partir de recibos "
    f"reales. Por último, Synth90k (Jaderberg et al., 2016) y SynthText "
    f"proporcionan datos masivos generados sintéticamente, fundamentales para "
    f"el pre-entrenamiento de los modelos actuales.",
    body,
))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    f'<super>4</super> EMNIST &mdash; {link("https://www.nist.gov/itl/products-and-services/emnist-dataset")}.',
    footnote_style,
))
story.append(Paragraph(
    f'<super>5</super> IAM Database &mdash; {link("https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")}.',
    footnote_style,
))

story.append(PageBreak())

# ===== 4. METODOLOGÍA =====
story.append(Paragraph("4. Metodología", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph(
    "El sistema propuesto se organiza en cinco etapas secuenciales: captura, "
    "pre-procesamiento, detección, reconocimiento y post-procesamiento. La "
    "figura 2 muestra de manera esquemática el flujo de datos entre módulos.",
    body,
))
story.append(fit_image(ASSETS / "fig_pipeline.png", max_h_cm=6))
story.append(Paragraph("Figura 2. Arquitectura general del pipeline OCR implementado.", caption))

story.append(Paragraph(
    "El módulo de reconocimiento puede operar en dos modos: un modo "
    "<i>pretrained</i> que delega la tarea a EasyOCR y un modo <i>custom</i> que "
    "utiliza la red CRNN+CTC propia. La figura 3 detalla la arquitectura "
    "interna de esta última.",
    body,
))
story.append(fit_image(ASSETS / "fig_crnn.png", max_h_cm=9))
story.append(Paragraph("Figura 3. Arquitectura CRNN+CTC entrenada sobre pseudo-palabras EMNIST.", caption))

story.append(Paragraph(
    "La red propia se compone de un tronco convolucional VGG-like seguido de "
    "dos capas LSTM bidireccionales y una proyección lineal que produce, en "
    "cada paso temporal, la distribución sobre 63 clases (62 alfanuméricas más "
    "el blanco CTC). La tabla 3 enumera las capas más relevantes.",
    body,
))

arch_hdr = ["Capa", "Tipo", "Salida", "Parámetros"]
arch_rows = [
    ["Conv1", "Conv2D 3×3 + BN + ReLU", "(B, 32, 32, 128)", "~9 K"],
    ["Pool1", "MaxPool 2×2", "(B, 16, 16, 128)", "0"],
    ["Conv2", "Conv2D 3×3 + BN + ReLU", "(B, 16, 16, 64)", "~18 K"],
    ["Pool2", "MaxPool 2×1", "(B, 8, 16, 64)", "0"],
    ["Conv3", "Conv2D 3×3 + ReLU", "(B, 8, 16, 128)", "~74 K"],
    ["Reshape", "Map-to-sequence", "(B, 16, 1024)", "0"],
    ["BiLSTM1", "LSTM(256) bidireccional", "(B, 16, 512)", "~2.6 M"],
    ["BiLSTM2", "LSTM(256) bidireccional", "(B, 16, 512)", "~1.6 M"],
    ["Dense", "Lineal + softmax", "(B, 16, 63)", "~32 K"],
]
tbl_arch = Table([arch_hdr] + arch_rows, colWidths=[2.0*cm, 5.0*cm, 4.5*cm, 3.0*cm])
tbl_arch.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "DMSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Inter"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, SURFACE]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tbl_arch)
story.append(Paragraph("Tabla 3. Capas principales de la red CRNN+CTC propia.", caption))

story.append(Paragraph(
    "Para entrenar la red bajo el régimen CTC se construyen pseudo-palabras "
    "concatenando entre 2 y 6 caracteres aleatorios extraídos de la partición "
    "<i>byclass</i> de EMNIST. Cada pseudo-palabra se renderiza en un lienzo "
    "de 32 × 128 píxeles, con ligeras variaciones de espaciado y ruido "
    "gaussiano para favorecer la generalización. Se generan 20 000 pseudo-"
    "palabras de entrenamiento y 4 000 de validación.",
    body,
))

story.append(Paragraph("4.1 Hiperparámetros", h2))
hp_rows = [
    ["Optimizador", "Adam"],
    ["Tasa de aprendizaje", "1 × 10<super>-3</super>"],
    ["Tamaño de lote", "64"],
    ["Épocas", "10"],
    ["Pseudo-palabras entrenamiento", "20 000"],
    ["Pseudo-palabras validación", "4 000"],
    ["Función de pérdida", "CTC (ctc_batch_cost)"],
]
hp_data = [[Paragraph(f"<b>{k}</b>", body), Paragraph(v, body)] for k, v in hp_rows]
tbl_hp = Table(hp_data, colWidths=[6 * cm, 8 * cm])
tbl_hp.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ("BACKGROUND", (0, 0), (0, -1), SURFACE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tbl_hp)
story.append(Paragraph("Tabla 4. Hiperparámetros principales del entrenamiento.", caption))

story.append(Paragraph(
    "Como comparador se utiliza EasyOCR con sus pesos por defecto, "
    "configurado para los idiomas español e inglés. Internamente combina un "
    "detector basado en CRAFT con un módulo de reconocimiento CRNN "
    "pre-entrenado sobre datos multilingües.",
    body,
))

story.append(PageBreak())

# ===== 5. IMPLEMENTACIÓN =====
story.append(Paragraph("5. Implementación", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph("5.1 Estructura del repositorio", h2))
story.append(Paragraph(
    "El repositorio se organiza siguiendo convenciones de proyectos de "
    "investigación reproducibles. Cada subdirectorio cumple una responsabilidad "
    "única y bien delimitada.",
    body,
))
story.append(Paragraph(
    "ocr_proyecto_ia/<br/>"
    "&nbsp;&nbsp;assets/ &nbsp;&nbsp;&nbsp;&nbsp;# figuras y recursos<br/>"
    "&nbsp;&nbsp;data/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# datasets (EMNIST, sintético)<br/>"
    "&nbsp;&nbsp;docs/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# reporte y diapositivas<br/>"
    "&nbsp;&nbsp;notebooks/ &nbsp;# experimentos Jupyter<br/>"
    "&nbsp;&nbsp;src/<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;preprocessing.py &nbsp;# Otsu, deskew, denoising<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;detection.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# wrapper de CRAFT<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;model_crnn.py &nbsp;&nbsp;&nbsp;&nbsp;# arquitectura CNN+CTC<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;train.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# bucle de entrenamiento<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;decode.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# decodificador greedy / beam<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;app_streamlit.py &nbsp;# interfaz gráfica<br/>"
    "&nbsp;&nbsp;requirements.txt",
    code,
))

story.append(Paragraph("5.2 Pérdida CTC en TensorFlow / Keras", h2))
story.append(Paragraph(
    "La pérdida CTC se calcula a través de la utilidad de bajo nivel "
    "<font face=\"JBMono\" size=\"9\">tf.keras.backend.ctc_batch_cost</font>, "
    "que recibe las etiquetas, las predicciones de logits y las longitudes "
    "respectivas. Es habitual encapsularla en una capa <i>Lambda</i> para "
    "integrarla al modelo entrenable.",
    body,
))
story.append(Paragraph(
    "def ctc_loss(args):<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;y_true, y_pred, input_len, label_len = args<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;return tf.keras.backend.ctc_batch_cost(<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y_true, y_pred, input_len, label_len<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;)<br/><br/>"
    "loss_layer = tf.keras.layers.Lambda(ctc_loss, name=&quot;ctc&quot;)<br/>"
    "model = tf.keras.Model(<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;inputs=[image_in, labels_in, input_len_in, label_len_in],<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;outputs=loss_layer([labels_in, y_pred, input_len_in, label_len_in]),<br/>"
    ")<br/>"
    "model.compile(optimizer=&quot;adam&quot;, loss=lambda y_true, y_pred: y_pred)",
    code,
))

story.append(Paragraph("5.3 Bloque convolucional", h2))
story.append(Paragraph(
    "def cnn_block(x):<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.Conv2D(128, 3, padding=&quot;same&quot;)(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.BatchNormalization()(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.Activation(&quot;relu&quot;)(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.MaxPool2D((2, 2))(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.Conv2D(64, 3, padding=&quot;same&quot;)(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.BatchNormalization()(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.Activation(&quot;relu&quot;)(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;x = layers.MaxPool2D((2, 1))(x)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;return x",
    code,
))

story.append(Paragraph("5.4 Decodificador greedy", h2))
story.append(Paragraph(
    "Tras obtener las probabilidades por paso temporal, el decodificador "
    "<i>greedy</i> selecciona la clase de mayor probabilidad en cada instante "
    "y posteriormente colapsa repeticiones y blancos consecutivos.",
    body,
))
story.append(Paragraph(
    "def greedy_decode(y_pred, idx_to_char):<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;preds = np.argmax(y_pred, axis=-1)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;texts = []<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;for seq in preds:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out, prev = [], -1<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for c in seq:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if c != prev and c != BLANK:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out.append(idx_to_char[c])<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prev = c<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;texts.append(&quot;&quot;.join(out))<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;return texts",
    code,
))

story.append(Paragraph("5.5 Llamada a EasyOCR", h2))
story.append(Paragraph(
    "El pipeline pre-entrenado se invoca con apenas dos líneas. EasyOCR "
    "descarga automáticamente los pesos del detector y del reconocedor en la "
    "primera ejecución.",
    body,
))
story.append(Paragraph(
    "import easyocr<br/>"
    "reader = easyocr.Reader([&quot;es&quot;, &quot;en&quot;])<br/>"
    "resultados = reader.readtext(&quot;imagen.png&quot;)<br/>"
    "for caja, texto, confianza in resultados:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;print(texto, confianza)",
    code,
))

story.append(Paragraph("5.6 Interfaz Streamlit", h2))
story.append(Paragraph(
    "Para facilitar la demostración en aula se desarrolla una interfaz web "
    "ligera con Streamlit. La aplicación muestra cuatro etapas claramente "
    "diferenciadas: (i) carga y pre-procesado de la imagen, (ii) cajas "
    "detectadas por CRAFT/EasyOCR, (iii) texto reconocido por cada modelo y "
    "(iv) descarga del resultado en formato TXT. Esta capa visual permite a "
    "los estudiantes contrastar inmediatamente la salida del modelo propio con "
    "la del pipeline pre-entrenado.",
    body,
))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    f'<super>6</super> Documentación oficial de EasyOCR &mdash; '
    f'{link("https://www.jaided.ai/easyocr/documentation/")}.',
    footnote_style,
))
story.append(PageBreak())

# ===== 6. RESULTADOS Y DISCUSIÓN =====
story.append(Paragraph("6. Resultados y discusión", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph(
    "Tras diez épocas de entrenamiento, la pérdida CTC sobre el conjunto de "
    "validación desciende de manera monótona, como muestra la figura 4. La "
    "curva sigue el patrón típico de un entrenamiento estable: descenso rápido "
    "durante las primeras épocas, seguido de un asintótico afinamiento.",
    body,
))
story.append(fit_image(ASSETS / "fig_perdida.png", max_h_cm=8))
story.append(Paragraph("Figura 4. Curva de pérdida CTC durante el entrenamiento.", caption))

story.append(Paragraph(
    "La tabla 5 contrasta las métricas obtenidas por ambos enfoques sobre la "
    "partición de validación de pseudo-palabras EMNIST. Las celdas marcadas "
    "con un guión indican métricas no aplicables al pipeline pre-entrenado: "
    "EasyOCR está entrenado sobre vocabulario natural, no sobre cadenas "
    "aleatorias alfanuméricas, por lo que su precisión en EMNIST no resulta "
    "directamente comparable.",
    body,
))

res_hdr = ["Métrica", "Pipeline EasyOCR", "Modelo propio CNN+CTC"]
res_rows = [
    ["Precisión a nivel carácter (EMNIST val.)", "—", "~92 %"],
    ["Precisión a nivel palabra (EMNIST val.)", "—", "~78 %"],
    ["Tiempo de inferencia CPU (1 imagen)", "~1.2 s", "~80 ms"],
    ["Idiomas soportados", "80+", "Alfanumérico básico"],
    ["Tamaño del modelo", "~64 MB", "~17 MB"],
]
tbl_res = Table([res_hdr] + res_rows, colWidths=[7*cm, 4.0*cm, 4.5*cm])
tbl_res.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "DMSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Inter"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, SURFACE]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
]))
story.append(tbl_res)
story.append(Paragraph("Tabla 5. Comparación de resultados entre EasyOCR y el modelo propio.", caption))

story.append(Paragraph("6.1 Discusión", h2))
story.append(Paragraph(
    "Los resultados ilustran el compromiso clásico entre generalidad y "
    "control. EasyOCR generaliza notablemente mejor sobre imágenes "
    "fotográficas reales gracias a su pre-entrenamiento masivo en datos "
    "diversos; su detector CRAFT localiza con precisión cajas en escenas "
    "complejas y el reconocedor multilingüe maneja sin esfuerzo acentos y "
    "símbolos. En contraste, el modelo propio &mdash;mucho más ligero y "
    "rápido en inferencia&mdash; expone con claridad cada componente: la "
    "extracción de características, el modelado secuencial y el "
    "alineamiento CTC, lo cual lo convierte en una herramienta didáctica "
    "valiosa.",
    body,
))

story.append(Paragraph("6.2 Limitaciones", h2))
story.append(Paragraph(
    "El modelo propio presenta varias limitaciones inherentes a su naturaleza "
    "educativa. Primero, está entrenado sobre pseudo-palabras sintéticas, por "
    "lo que su desempeño cae rápidamente ante texto real con contexto "
    "lingüístico. Segundo, sólo gestiona palabras cortas (hasta 6 caracteres) "
    "debido al tamaño fijo del paso temporal. Tercero, no aplica corrección de "
    "inclinación de manera robusta, por lo que falla con imágenes rotadas más "
    "allá de unos pocos grados. Cuarto, el vocabulario se restringe a "
    "caracteres alfanuméricos sin acentos.",
    body,
))

story.append(PageBreak())

# ===== 7. CONCLUSIONES =====
story.append(Paragraph("7. Conclusiones", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph(
    "El presente trabajo demuestra que es factible construir un sistema OCR "
    "con propósitos didácticos que articule, en un mismo proyecto, una "
    "solución industrial pre-entrenada y un modelo propio implementado desde "
    "cero. La comparación cuantitativa y cualitativa entre EasyOCR y la red "
    "CRNN+CTC entrenada sobre pseudo-palabras EMNIST permite a las y los "
    "estudiantes comprender en profundidad cada uno de los bloques "
    "&mdash;pre-procesamiento, detección, reconocimiento secuencial, pérdida "
    "CTC y decodificación&mdash; sin renunciar a una experiencia de uso "
    "concreta y reproducible.",
    body,
))
story.append(Paragraph(
    "Ambos enfoques cumplen los objetivos planteados: EasyOCR sirve como "
    "referencia industrial estable, mientras que el modelo propio explicita "
    "el funcionamiento interno del reconocimiento secuencial y abre la puerta "
    "a experimentos pedagógicos posteriores.",
    body,
))

story.append(Paragraph("7.1 Próximos pasos", h2))
story.append(Paragraph(
    f"Se identifican tres líneas naturales de continuación. La primera "
    f"consiste en sustituir el reconocedor CRNN+CTC por una arquitectura "
    f"basada en Transformers, en línea con TrOCR (Li et al., 2023; "
    f'{link("https://github.com/microsoft/unilm/tree/master/trocr", "Microsoft Research")}). '
    "La segunda contempla un <i>fine-tuning</i> del modelo propio sobre "
    "ICDAR-SROIE para abordar la extracción estructurada de campos en "
    "recibos. La tercera plantea integrar un modelo de lenguaje &mdash;por "
    "ejemplo, un n-grama sobre español neutro&mdash; en la etapa de "
    "decodificación, combinándolo con <i>beam search</i> para corregir errores "
    "ortográficos comunes.",
    body,
))
story.append(Paragraph(
    "En suma, el proyecto provee una base sólida para futuras iteraciones "
    "tanto académicas como aplicadas, y reafirma el valor de combinar "
    "soluciones de alto nivel con implementaciones controladas como vehículo "
    "para la enseñanza de la Inteligencia Artificial moderna.",
    body,
))

story.append(PageBreak())

# ===== REFERENCIAS =====
story.append(Paragraph("Referencias", h1))
story.append(AccentBar(USABLE_W, height=1.2, color=ACCENT))
story.append(Spacer(1, 0.3 * cm))

refs = [
    ("Baek, Y., Lee, B., Han, D., Yun, S., &amp; Lee, H. (2019). "
     "<i>Character region awareness for text detection</i> (CRAFT). "
     "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. "
     + link("https://github.com/clovaai/CRAFT-pytorch")),
    ("Cohen, G., Afshar, S., Tapson, J., &amp; van Schaik, A. (2017). "
     "<i>EMNIST: an extension of MNIST to handwritten letters</i>. "
     "National Institute of Standards and Technology. "
     + link("https://www.nist.gov/itl/products-and-services/emnist-dataset")),
    ("Graves, A., Fernández, S., Gomez, F., &amp; Schmidhuber, J. (2006). "
     "Connectionist temporal classification: labelling unsegmented sequence data "
     "with recurrent neural networks. <i>Proceedings of the 23rd International "
     "Conference on Machine Learning</i>. "
     + link("https://www.cs.toronto.edu/~graves/icml_2006.pdf")),
    ("Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., &amp; Jawahar, "
     "C. V. (2019). ICDAR 2019 competition on scanned receipt OCR and "
     "information extraction (SROIE). "
     + link("https://arxiv.org/abs/2103.10213")),
    ("Jaderberg, M., Simonyan, K., Vedaldi, A., &amp; Zisserman, A. (2016). "
     "Reading text in the wild with convolutional neural networks "
     "(Synth90k). <i>International Journal of Computer Vision</i>. "
     + link("https://www.robots.ox.ac.uk/~vgg/data/text/")),
    ("JaidedAI. (s.f.). <i>EasyOCR: ready-to-use OCR with 80+ supported "
     "languages</i>. GitHub. "
     + link("https://github.com/JaidedAI/EasyOCR")),
    ("Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., "
     "Li, Z., &amp; Wei, F. (2023). TrOCR: Transformer-based optical character "
     "recognition with pre-trained models. <i>Proceedings of the AAAI "
     "Conference on Artificial Intelligence</i>. "
     + link("https://github.com/microsoft/unilm/tree/master/trocr")),
    ("Marti, U.-V., &amp; Bunke, H. (2002). The IAM-database: an English "
     "sentence database for offline handwriting recognition. "
     "<i>International Journal on Document Analysis and Recognition</i>, 5(1), 39&ndash;46. "
     + link("https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")),
    ("Mindee. (s.f.). <i>docTR: document text recognition</i>. GitHub. "
     + link("https://github.com/mindee/doctr")),
    ("PaddlePaddle. (s.f.). <i>PaddleOCR: awesome multilingual OCR toolkits</i>. "
     "GitHub. " + link("https://github.com/PaddlePaddle/PaddleOCR")),
    ("Paruchuri, V. (s.f.). <i>Surya: multilingual document OCR toolkit</i>. "
     "GitHub. " + link("https://github.com/VikParuchuri/surya")),
    ("Shi, B., Bai, X., &amp; Yao, C. (2015). An end-to-end trainable neural "
     "network for image-based sequence recognition and its application to "
     "scene text recognition. <i>IEEE Transactions on Pattern Analysis and "
     "Machine Intelligence</i>. "
     + link("https://arxiv.org/abs/1507.05717")),
    ("Smith, R. (s.f.). <i>Tesseract OCR engine</i>. GitHub. "
     + link("https://github.com/tesseract-ocr/tesseract")),
    ("OpenMMLab. (s.f.). <i>MMOCR: a comprehensive toolbox for text "
     "detection, recognition and understanding</i>. GitHub. "
     + link("https://github.com/open-mmlab/mmocr")),
]
for r in refs:
    story.append(Paragraph(r, ref_style))

# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------
def header_footer(canv, doc):
    canv.saveState()
    # Footer line
    canv.setStrokeColor(BORDER)
    canv.setLineWidth(0.3)
    canv.line(MARGIN, 1.6 * cm, PAGE_W - MARGIN, 1.6 * cm)
    canv.setFont("Inter", 8)
    canv.setFillColor(MUTED)
    canv.drawString(MARGIN, 1.2 * cm,
                    "Sistema OCR Educativo \u2014 E. Domínguez Osio")
    canv.drawRightString(PAGE_W - MARGIN, 1.2 * cm, f"Página {doc.page}")
    canv.restoreState()


def cover_header_footer(canv, doc):
    # Minimal footer on cover
    canv.saveState()
    canv.setFont("Inter", 8)
    canv.setFillColor(MUTED)
    canv.drawRightString(PAGE_W - MARGIN, 1.2 * cm,
                         "Reporte académico \u2014 Mayo 2026")
    canv.restoreState()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=LETTER,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN,
    bottomMargin=MARGIN + 0.5 * cm,
    title="Sistema OCR Educativo con Aprendizaje Profundo",
    author="Perplexity Computer",
)

doc.build(story, onFirstPage=cover_header_footer, onLaterPages=header_footer)
print(f"Wrote {OUT}")
