# api.py
import os
import uuid
import tempfile
import time
import locale
from datetime import datetime
from typing import Optional, Dict
import shutil  # <--- nuevo

# --- Matplotlib en modo no-interactivo (Railway/servers headless) ---
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import google.generativeai as genai

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fastapi import FastAPI, HTTPException, UploadFile, File  # <--- agregado UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

# --------------------------------------------------------------------------------------
# Configuración y utilidades
# --------------------------------------------------------------------------------------

# CSV fijo dentro del repo (no se sube en la request) — lo dejamos como fallback opcional
DEFAULT_INPUT_CSV_FILE = "reporte_redes_sociales.csv"

# Carpeta temporal para reportes (no persiste entre despliegues)
REPORTS_DIR = os.path.join(tempfile.gettempdir(), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Locale para español (si no existe, cae a inglés)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    pass

def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Falta variable de entorno GEMINI_API_KEY.")
    return key

# --------------------------------------------------------------------------------------
# Llamada a IA (Gemini)
# --------------------------------------------------------------------------------------
def ia_call_with_retry(prompt: str, valid_responses: Optional[list[str]] = None) -> str:
    try:
        genai.configure(api_key=get_api_key())
        model = genai.GenerativeModel('models/gemini-2.5-pro')

        for _ in range(3):
            response = model.generate_content(prompt)
            text = (getattr(response, "text", "") or "").strip()
            if valid_responses:
                if text in valid_responses:
                    return text
            else:
                return text
            time.sleep(0.8)

        return "Indeterminado" if valid_responses else ""
    except Exception as e:
        print(f"[WARN] Error en API de Gemini: {e}")
        return "Indeterminado" if valid_responses else ""

# --------------------------------------------------------------------------------------
# Lógica de negocio
# --------------------------------------------------------------------------------------
def analizar_sentimiento_comentario(comments_string):
    if not comments_string or pd.isna(comments_string):
        return "Neutro"
    prompt = (
        'Analiza el sentimiento de los siguientes comentarios. '
        'Clasifícalo como "Positivo", "Negativo" o "Neutro". '
        'Responde solo con una de esas tres palabras.\n\n'
        f'Comentarios:\n"{comments_string}"'
    )
    return ia_call_with_retry(prompt, ["Positivo", "Negativo", "Neutro"])


def crear_graficos(df: pd.DataFrame):
    print("Creando gráficos...")
    sns.set_style("whitegrid")

    # Gráfico 1: Distribución de Sentimiento
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette("viridis", len(sentiment_counts)),
        startangle=90
    )
    plt.title('Distribución General del Sentimiento', fontsize=16, pad=20)
    plt.savefig('grafico_sentimiento_distribucion.png', dpi=300)
    print("-> Gráfico 'grafico_sentimiento_distribucion.png' guardado.")

    # Gráfico 2: Sentimiento por Plataforma
    sentiment_by_platform = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_platform.plot(
        kind='bar',
        figsize=(12, 7),
        color=sns.color_palette("magma", len(sentiment_counts))
    )
    plt.title('Volumen de Sentimiento por Plataforma', fontsize=16, pad=20)
    plt.xlabel('Plataforma', fontsize=12)
    plt.ylabel('Cantidad de Publicaciones', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Sentimiento')
    plt.tight_layout()
    plt.savefig('grafico_sentimiento_por_plataforma.png', dpi=300)
    print("-> Gráfico 'grafico_sentimiento_por_plataforma.png' guardado.")

    # Gráfico 3: Sentimiento por Semana
    df['week'] = df['created_date'].dt.strftime('Semana %U')
    sentiment_by_week = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_week.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 7),
        color=sns.color_palette("plasma", len(sentiment_counts))
    )
    plt.title('Evolución Semanal del Sentimiento', fontsize=16, pad=20)
    plt.xlabel('Semana del Mes', fontsize=12)
    plt.ylabel('Cantidad de Publicaciones', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentimiento')
    plt.tight_layout()
    plt.savefig('grafico_sentimiento_semanal.png', dpi=300)
    print("-> Gráfico 'grafico_sentimiento_semanal.png' guardado.")

    plt.close('all')


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.cell(0, 10, 'CONFIDENCIAL | REPORTE ESTRATÉGICO DE REDES SOCIALES', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 10, 'Lider Bci', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(10, 20, 200, 20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

    def create_title_page(self, title, subtitle, date_range):
        self.add_page()
        self.set_y(80)
        self.set_font('Helvetica', 'B', 28)
        self.multi_cell(0, 15, title, align='C')
        self.ln(10)
        self.set_font('Helvetica', '', 18)
        self.multi_cell(0, 12, subtitle, align='C')
        self.ln(20)
        self.set_font('Helvetica', 'I', 12)
        self.multi_cell(0, 10, f"Período de Análisis: {date_range}", align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, f' {title}', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        safe = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 7, safe)
        self.ln()


def generar_reporte_completo(input_csv_path: str, output_pdf_path: str) -> str:
    print("Iniciando generación de Reporte Estratégico Mensual...")

    if not os.path.exists(input_csv_path):
        raise ValueError(f"No existe el CSV: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    if 'created_date' not in df.columns:
        raise ValueError("El CSV no contiene la columna 'created_date'.")

    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df = df.dropna(subset=['created_date'])

    now = datetime.now()
    df_mes_actual = df[(df['created_date'].dt.month == now.month) & (df['created_date'].dt.year == now.year)].copy()

    if df_mes_actual.empty:
        raise ValueError(f"No se encontraron datos para {now.strftime('%B %Y')}.")

    analysis_period = now.strftime('%B %Y').capitalize()
    print(f"Analizando {len(df_mes_actual)} publicaciones de {analysis_period}.")

    # IA: sentimiento
    print("IA: Analizando Sentimiento...")
    df_mes_actual['sentiment'] = df_mes_actual['post_comments'].apply(analizar_sentimiento_comentario)

    # KPIs
    print("Calculando KPIs y resúmenes para la IA...")
    total_posts = len(df_mes_actual)
    total_likes = df_mes_actual['likes_reactions'].sum()
    total_comments = df_mes_actual['comments_count'].sum()
    total_shares = df_mes_actual['shares'].sum()

    kpis_generales_str = (
        f"- Total de Publicaciones: {total_posts}\n"
        f"- Total de Likes/Reacciones: {total_likes}\n"
        f"- Total de Comentarios: {total_comments}\n"
        f"- Total de Shares: {total_shares}"
    )

    sentiment_counts = df_mes_actual['sentiment'].value_counts()
    sentimiento_general_str = sentiment_counts.to_string()

    df_mes_actual['week'] = df_mes_actual['created_date'].dt.strftime('Semana %U')
    sentimiento_semanal = pd.crosstab(df_mes_actual['week'], df_mes_actual['sentiment'])
    sentimiento_semanal_str = sentimiento_semanal.to_string()

    sentimiento_plataforma = pd.crosstab(df_mes_actual['platform'], df_mes_actual['sentiment'])
    sentimiento_plataforma_str = sentimiento_plataforma.to_string()

    crear_graficos(df_mes_actual)

    print("IA: Generando análisis estratégico...")
    main_prompt = f"""
Rol: Actúa como un Analista Senior de Estrategia Digital para la marca "Lider Bci". Tu objetivo es crear un informe mensual conciso y accionable para la gerencia, transformando datos en insights y una hoja de ruta clara.
Contexto: Has recibido un resumen de KPIs y datos de sentimiento de redes sociales para el período de {analysis_period}. No necesitas recalcular nada; tu tarea es interpretar estos resúmenes y conectar los puntos.
---
DATOS PRE-PROCESADOS DEL MES:

**KPIs Generales:**
{kpis_generales_str}

**Distribución General de Sentimiento (Cantidad de publicaciones):**
{sentimiento_general_str}

**Análisis de Sentimiento Semanal (Cantidad de publicaciones):**
{sentimiento_semanal_str}

**Análisis de Sentimiento por Plataforma (Cantidad de publicaciones):**
{sentimiento_plataforma_str}
---
**Tu Tarea - Estructura del Informe:**
Basado ESTRICTAMENTE en los datos resumidos de arriba, genera el informe siguiendo esta estructura:

**1. Conclusión General y KPIs Clave:**
- Inicia con un párrafo ejecutivo (3-4 líneas) que resuma el estado del mes. ¿Fue un mes positivo, negativo o de transición? ¿Cuál es el insight más importante?
- Presenta una tabla en Markdown con los KPIs Generales y la Distribución General de Sentimiento.

**2. Análisis de Sentimiento Detallado:**
- **Por Semana:** Analiza la tabla de sentimiento semanal. ¿Hubo alguna semana con un pico de comentarios negativos o positivos? Hipotetiza brevemente por qué pudo haber ocurrido.
- **Por Plataforma:** Analiza la tabla de sentimiento por plataforma. ¿Qué canal tiene una conversación más saludable? ¿Dónde se concentran los comentarios negativos?

**3. Propuestas Estratégicas para Mejorar (5 Iniciativas):**
- Propón exactamente 5 iniciativas concretas y accionables para el próximo mes. Cada propuesta debe ser clara y tener un objetivo definido. Estructúralas así:
    - **Iniciativa 1 (Capitalizar Sentimiento Positivo):** ...
    - **Iniciativa 2 (Mitigar Riesgo Negativo):** ...
    - **Iniciativa 3 (Optimización de Contenido):** ...
    - **Iniciativa 4 (Fomento de Comunidad):** ...
    - **Iniciativa 5 (Estrategia de Plataforma):** ...

Usa Markdown para encabezados y tablas.
    """.strip()

    reporte_texto = ia_call_with_retry(main_prompt)

    print("Ensamblando PDF final...")
    pdf = PDF('P', 'mm', 'A4')
    pdf.create_title_page('Reporte Estratégico Mensual', 'Inteligencia de Redes Sociales', analysis_period)

    pdf.add_page()
    pdf.chapter_title('Análisis y Propuestas Estratégicas')
    pdf.chapter_body(reporte_texto)

    pdf.add_page()
    pdf.chapter_title('Apéndice Visual')
    graficos = [
        ('grafico_sentimiento_distribucion.png', 'Distribución General de Sentimiento'),
        ('grafico_sentimiento_por_plataforma.png', 'Volumen de Sentimiento por Plataforma'),
        ('grafico_sentimiento_semanal.png', 'Evolución Semanal del Sentimiento'),
    ]
    for path, title in graficos:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, title, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        pdf.image(path, x=15, w=180)

    pdf.output(output_pdf_path)
    print(f"¡Éxito! Reporte guardado como '{output_pdf_path}'")
    return output_pdf_path

# --------------------------------------------------------------------------------------
# API (FastAPI)
# --------------------------------------------------------------------------------------
api = FastAPI(title="RRSS Reporte API", version="1.0.0")

# Memoria simple en proceso: report_id -> path
REPORT_INDEX: Dict[str, str] = {}

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/run")
async def run_report(csv_file: UploadFile = File(...)):
    """
    Sube un CSV (multipart/form-data) y genera el PDF.
    Columnas requeridas:
      created_date, platform, post_comments, likes_reactions, comments_count, shares
    """
    # 1) Validar extensión
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .csv")

    # 2) Guardar CSV subido a un path temporal
    temp_csv_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.csv")
    try:
        with open(temp_csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el CSV subido: {e}")

    # 3) Validación rápida de columnas esperadas (error legible antes de correr el pipeline)
    required_cols = {"created_date", "platform", "post_comments", "likes_reactions", "comments_count", "shares"}
    try:
        df_head = pd.read_csv(temp_csv_path, nrows=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {e}")

    missing = [c for c in required_cols if c not in df_head.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas en el CSV: {missing}")

    # 4) Preparar salida PDF
    report_id = uuid.uuid4().hex
    outfile = f"Reporte_Estrategico_Mensual_{report_id}.pdf"
    out_path = os.path.join(REPORTS_DIR, outfile)

    # 5) Ejecutar pipeline
    try:
        generar_reporte_completo(temp_csv_path, out_path)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {e}")

    # 6) Indexar y responder
    REPORT_INDEX[report_id] = out_path
    return JSONResponse(
        {
            "report_id": report_id,
            "filename": os.path.basename(out_path),
            "download_url": f"/download/{report_id}"
        }
    )

@api.get("/download/{report_id}")
def download_report(report_id: str):
    """
    Descarga el PDF generado previamente con /run.
    """
    path = REPORT_INDEX.get(report_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado o expirado.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
