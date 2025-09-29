# # api.py
# import os
# import uuid
# import tempfile
# import time
# import locale
# from datetime import datetime
# from typing import Optional, Dict
# import shutil

# # --- Matplotlib en modo no-interactivo (Railway/servers headless) ---
# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import google.generativeai as genai

# from fpdf import FPDF
# from fpdf.enums import XPos, YPos
# from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse

# # --------------------------------------------------------------------------------------
# # Configuración y utilidades
# # --------------------------------------------------------------------------------------

# DEFAULT_INPUT_CSV_FILE = "reporte_redes_sociales.csv"

# REPORTS_DIR = os.path.join(tempfile.gettempdir(), "reports")
# os.makedirs(REPORTS_DIR, exist_ok=True)

# # Para silenciar algunos warnings de gRPC (opcional)
# os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# try:
#     locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
# except locale.Error:
#     pass

# def get_api_key() -> str:
#     key = os.getenv("GEMINI_API_KEY", "").strip()
#     if not key:
#         raise RuntimeError("Falta variable de entorno GEMINI_API_KEY.")
#     return key

# # --------------------------------------------------------------------------------------
# # Llamada a IA (Gemini)
# # --------------------------------------------------------------------------------------
# def ia_call_with_retry(prompt: str, valid_responses: Optional[list[str]] = None) -> str:
#     try:
#         genai.configure(api_key=get_api_key())
#         model = genai.GenerativeModel('models/gemini-2.5-pro')

#         for _ in range(3):
#             response = model.generate_content(prompt)
#             text = (getattr(response, "text", "") or "").strip()
#             if valid_responses:
#                 if text in valid_responses:
#                     return text
#             else:
#                 return text
#             time.sleep(0.8)

#         return "Indeterminado" if valid_responses else ""
#     except Exception as e:
#         print(f"[WARN] Error en API de Gemini: {e}")
#         return "Indeterminado" if valid_responses else ""

# # --------------------------------------------------------------------------------------
# # Lógica de negocio
# # --------------------------------------------------------------------------------------
# def analizar_sentimiento_comentario(comments_string):
#     if not comments_string or pd.isna(comments_string):
#         return "Neutro"
#     prompt = (
#         'Analiza el sentimiento de los siguientes comentarios. '
#         'Clasifícalo como "Positivo", "Negativo" o "Neutro". '
#         'Responde solo con una de esas tres palabras.\n\n'
#         f'Comentarios:\n"{comments_string}"'
#     )
#     return ia_call_with_retry(prompt, ["Positivo", "Negativo", "Neutro"])


# def crear_graficos(df: pd.DataFrame):
#     print("Creando gráficos...")
#     sns.set_style("whitegrid")

#     # Gráfico 1: Distribución de Sentimiento
#     sentiment_counts = df['sentiment'].value_counts()
#     plt.figure(figsize=(8, 8))
#     plt.pie(
#         sentiment_counts,
#         labels=sentiment_counts.index,
#         autopct='%1.1f%%',
#         colors=sns.color_palette("viridis", len(sentiment_counts)),
#         startangle=90
#     )
#     plt.title('Distribución General del Sentimiento', fontsize=16, pad=20)
#     plt.savefig('grafico_sentimiento_distribucion.png', dpi=300)
#     print("-> Gráfico 'grafico_sentimiento_distribucion.png' guardado.")

#     # Gráfico 2: Sentimiento por Plataforma
#     sentiment_by_platform = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
#     sentiment_by_platform.plot(
#         kind='bar',
#         figsize=(12, 7),
#         color=sns.color_palette("magma", len(sentiment_counts))
#     )
#     plt.title('Volumen de Sentimiento por Plataforma', fontsize=16, pad=20)
#     plt.xlabel('Plataforma', fontsize=12)
#     plt.ylabel('Cantidad de Publicaciones', fontsize=12)
#     plt.xticks(rotation=0)
#     plt.legend(title='Sentimiento')
#     plt.tight_layout()
#     plt.savefig('grafico_sentimiento_por_plataforma.png', dpi=300)
#     print("-> Gráfico 'grafico_sentimiento_por_plataforma.png' guardado.")

#     # Gráfico 3: Sentimiento por Semana
#     df['week'] = df['created_date'].dt.strftime('Semana %U')
#     sentiment_by_week = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
#     sentiment_by_week.plot(
#         kind='bar',
#         stacked=True,
#         figsize=(12, 7),
#         color=sns.color_palette("plasma", len(sentiment_counts))
#     )
#     plt.title('Evolución Semanal del Sentimiento', fontsize=16, pad=20)
#     plt.xlabel('Semana del Mes', fontsize=12)
#     plt.ylabel('Cantidad de Publicaciones', fontsize=12)
#     plt.xticks(rotation=45, ha='right')
#     plt.legend(title='Sentimiento')
#     plt.tight_layout()
#     plt.savefig('grafico_sentimiento_semanal.png', dpi=300)
#     print("-> Gráfico 'grafico_sentimiento_semanal.png' guardado.")

#     plt.close('all')


# class PDF(FPDF):
#     def header(self):
#         self.set_font('Helvetica', 'B', 8)
#         self.cell(0, 10, 'CONFIDENCIAL | REPORTE ESTRATÉGICO DE REDES SOCIALES', new_x=XPos.RIGHT, new_y=YPos.TOP)
#         self.cell(0, 10, 'Lider Bci', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
#         self.line(10, 20, 200, 20)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Helvetica', 'I', 8)
#         self.cell(0, 10, f'Página {self.page_no()}', align='C')

#     def create_title_page(self, title, subtitle, date_range):
#         self.add_page()
#         self.set_y(80)
#         self.set_font('Helvetica', 'B', 28)
#         self.multi_cell(0, 15, title, align='C')
#         self.ln(10)
#         self.set_font('Helvetica', '', 18)
#         self.multi_cell(0, 12, subtitle, align='C')
#         self.ln(20)
#         self.set_font('Helvetica', 'I', 12)
#         self.multi_cell(0, 10, f"Período de Análisis: {date_range}", align='C')

#     def chapter_title(self, title):
#         self.set_font('Helvetica', 'B', 14)
#         self.set_fill_color(230, 230, 230)
#         self.cell(0, 10, f' {title}', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
#         self.ln(5)

#     def chapter_body(self, body):
#         self.set_font('Helvetica', '', 11)
#         safe = body.encode('latin-1', 'replace').decode('latin-1')
#         self.multi_cell(0, 7, safe)
#         self.ln()


# def generar_reporte_completo(input_csv_path: str, output_pdf_path: str) -> str:
#     print("Iniciando generación de Reporte Estratégico Mensual...")

#     if not os.path.exists(input_csv_path):
#         raise ValueError(f"No existe el CSV: {input_csv_path}")

#     df = pd.read_csv(input_csv_path)
#     if 'created_date' not in df.columns:
#         raise ValueError("El CSV no contiene la columna 'created_date'.")

#     df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
#     df = df.dropna(subset=['created_date'])

#     now = datetime.now()
#     df_mes_actual = df[(df['created_date'].dt.month == now.month) & (df['created_date'].dt.year == now.year)].copy()

#     if df_mes_actual.empty:
#         raise ValueError(f"No se encontraron datos para {now.strftime('%B %Y')}.")

#     analysis_period = now.strftime('%B %Y').capitalize()
#     print(f"Analizando {len(df_mes_actual)} publicaciones de {analysis_period}.")

#     # IA: sentimiento
#     print("IA: Analizando Sentimiento...")
#     df_mes_actual['sentiment'] = df_mes_actual['post_comments'].apply(analizar_sentimiento_comentario)

#     # KPIs
#     print("Calculando KPIs y resúmenes para la IA...")
#     total_posts = len(df_mes_actual)
#     total_likes = df_mes_actual['likes_reactions'].sum()
#     total_comments = df_mes_actual['comments_count'].sum()
#     total_shares = df_mes_actual['shares'].sum()

#     kpis_generales_str = (
#         f"- Total de Publicaciones: {total_posts}\n"
#         f"- Total de Likes/Reacciones: {total_likes}\n"
#         f"- Total de Comentarios: {total_comments}\n"
#         f"- Total de Shares: {total_shares}"
#     )

#     sentiment_counts = df_mes_actual['sentiment'].value_counts()
#     sentimiento_general_str = sentiment_counts.to_string()

#     df_mes_actual['week'] = df_mes_actual['created_date'].dt.strftime('Semana %U')
#     sentimiento_semanal = pd.crosstab(df_mes_actual['week'], df_mes_actual['sentiment'])
#     sentimiento_semanal_str = sentimiento_semanal.to_string()

#     sentimiento_plataforma = pd.crosstab(df_mes_actual['platform'], df_mes_actual['sentiment'])
#     sentimiento_plataforma_str = sentimiento_plataforma.to_string()

#     crear_graficos(df_mes_actual)

#     print("IA: Generando análisis estratégico...")
#     main_prompt = f"""
# Rol: Actúa como un Analista Senior de Estrategia Digital para la marca "Lider Bci". Tu objetivo es crear un informe mensual conciso y accionable para la gerencia, transformando datos en insights y una hoja de ruta clara.
# Contexto: Has recibido un resumen de KPIs y datos de sentimiento de redes sociales para el período de {analysis_period}. No necesitas recalcular nada; tu tarea es interpretar estos resúmenes y conectar los puntos.
# ---
# DATOS PRE-PROCESADOS DEL MES:

# **KPIs Generales:**
# {kpis_generales_str}

# **Distribución General de Sentimiento (Cantidad de publicaciones):**
# {sentimiento_general_str}

# **Análisis de Sentimiento Semanal (Cantidad de publicaciones):**
# {sentimiento_semanal_str}

# **Análisis de Sentimiento por Plataforma (Cantidad de publicaciones):**
# {sentimiento_plataforma_str}
# ---
# **Tu Tarea - Estructura del Informe:**
# Basado ESTRICTAMENTE en los datos resumidos de arriba, genera el informe siguiendo esta estructura:

# **1. Conclusión General y KPIs Clave:**
# - Inicia con un párrafo ejecutivo (3-4 líneas) que resuma el estado del mes. ¿Fue un mes positivo, negativo o de transición? ¿Cuál es el insight más importante?
# - Presenta una tabla en Markdown con los KPIs Generales y la Distribución General de Sentimiento.

# **2. Análisis de Sentimiento Detallado:**
# - **Por Semana:** Analiza la tabla de sentimiento semanal. ¿Hubo alguna semana con un pico de comentarios negativos o positivos? Hipotetiza brevemente por qué pudo haber ocurrido.
# - **Por Plataforma:** Analiza la tabla de sentimiento por plataforma. ¿Qué canal tiene una conversación más saludable? ¿Dónde se concentran los comentarios negativos?

# **3. Propuestas Estratégicas para Mejorar (5 Iniciativas):**
# - Propón exactamente 5 iniciativas concretas y accionables para el próximo mes. Cada propuesta debe ser clara y tener un objetivo definido. Estructúralas así:
#     - **Iniciativa 1 (Capitalizar Sentimiento Positivo):** ...
#     - **Iniciativa 2 (Mitigar Riesgo Negativo):** ...
#     - **Iniciativa 3 (Optimización de Contenido):** ...
#     - **Iniciativa 4 (Fomento de Comunidad):** ...
#     - **Iniciativa 5 (Estrategia de Plataforma):** ...

# Usa Markdown para encabezados y tablas.
#     """.strip()

#     reporte_texto = ia_call_with_retry(main_prompt)

#     print("Ensamblando PDF final...")
#     pdf = PDF('P', 'mm', 'A4')
#     pdf.create_title_page('Reporte Estratégico Mensual', 'Inteligencia de Redes Sociales', analysis_period)

#     pdf.add_page()
#     pdf.chapter_title('Análisis y Propuestas Estratégicas')
#     pdf.chapter_body(reporte_texto)

#     pdf.add_page()
#     pdf.chapter_title('Apéndice Visual')
#     graficos = [
#         ('grafico_sentimiento_distribucion.png', 'Distribución General de Sentimiento'),
#         ('grafico_sentimiento_por_plataforma.png', 'Volumen de Sentimiento por Plataforma'),
#         ('grafico_sentimiento_semanal.png', 'Evolución Semanal del Sentimiento'),
#     ]
#     for path, title in graficos:
#         pdf.set_font('Helvetica', 'B', 12)
#         pdf.cell(0, 10, title, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
#         pdf.ln(5)
#         pdf.image(path, x=15, w=180)

#     pdf.output(output_pdf_path)
#     print(f"¡Éxito! Reporte guardado como '{output_pdf_path}'")
#     return output_pdf_path

# # --------------------------------------------------------------------------------------
# # API (FastAPI)
# # --------------------------------------------------------------------------------------
# api = FastAPI(title="RRSS Reporte API", version="1.0.0")

# # Memorias en proceso
# REPORT_INDEX: Dict[str, str] = {}   # report_id -> pdf path
# REPORT_STATUS: Dict[str, str] = {}  # report_id -> "pending" | "ready" | "error"

# @api.get("/health")
# def health():
#     return {"status": "ok"}

# def _pipeline_background(csv_path: str, out_path: str, report_id: str):
#     try:
#         REPORT_STATUS[report_id] = "pending"
#         generar_reporte_completo(csv_path, out_path)
#         REPORT_INDEX[report_id] = out_path
#         REPORT_STATUS[report_id] = "ready"
#     except Exception as e:
#         print(f"[BG ERROR] {e}")
#         REPORT_STATUS[report_id] = "error"

# @api.post("/run")
# async def run_report(background_tasks: BackgroundTasks, csv_file: UploadFile = File(...)):
#     """
#     Recibe un CSV (multipart/form-data) y encola la generación del PDF en background.
#     Columnas requeridas: created_date, platform, post_comments, likes_reactions, comments_count, shares
#     """
#     # 1) Validar extensión
#     if not csv_file.filename.lower().endswith(".csv"):
#         raise HTTPException(status_code=400, detail="El archivo debe ser .csv")

#     # 2) Guardar CSV subido
#     temp_csv_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.csv")
#     try:
#         with open(temp_csv_path, "wb") as f:
#             shutil.copyfileobj(csv_file.file, f)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"No se pudo guardar el CSV subido: {e}")

#     # 3) Validar columnas requeridas
#     required_cols = {"created_date", "platform", "post_comments", "likes_reactions", "comments_count", "shares"}
#     try:
#         df_head = pd.read_csv(temp_csv_path, nrows=5)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {e}")

#     missing = [c for c in required_cols if c not in df_head.columns]
#     if missing:
#         raise HTTPException(status_code=400, detail=f"Faltan columnas en el CSV: {missing}")

#     # 4) Preparar salida
#     report_id = uuid.uuid4().hex
#     outfile = f"Reporte_Estrategico_Mensual_{report_id}.pdf"
#     out_path = os.path.join(REPORTS_DIR, outfile)

#     # 5) Encolar tarea en background y responder de inmediato
#     REPORT_STATUS[report_id] = "pending"
#     background_tasks.add_task(_pipeline_background, temp_csv_path, out_path, report_id)

#     return JSONResponse(
#         status_code=202,
#         content={
#             "report_id": report_id,
#             "status_url": f"/status/{report_id}",
#             "download_url": f"/download/{report_id}"
#         }
#     )

# @api.get("/status/{report_id}")
# def report_status(report_id: str):
#     status = REPORT_STATUS.get(report_id)
#     if not status:
#         raise HTTPException(status_code=404, detail="Reporte no encontrado.")
#     payload = {"report_id": report_id, "status": status}
#     if status == "ready" and report_id in REPORT_INDEX:
#         payload["filename"] = os.path.basename(REPORT_INDEX[report_id])
#         payload["download_url"] = f"/download/{report_id}"
#     return JSONResponse(payload)

# @api.get("/download/{report_id}")
# def download_report(report_id: str):
#     """
#     Descarga el PDF generado previamente con /run.
#     """
#     path = REPORT_INDEX.get(report_id)
#     if not path or not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Reporte no encontrado o expirado.")
#     return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))

import os
import uuid
import tempfile
import time
import locale
import shutil
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# --- Matplotlib en modo no-interactivo (Railway/servers headless) ---
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
import pytz
import google.generativeai as genai

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator

# --------------------------------------------------------------------------------------
# Configuración y utilidades
# --------------------------------------------------------------------------------------

API_VERSION = "v19.0"
POST_LIMIT_DEFAULT = 50
COMMENT_LIMIT_DEFAULT = 100

REPORTS_DIR = os.path.join(tempfile.gettempdir(), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Para silenciar algunos warnings de gRPC (opcional)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# Locale para español (si no existe, cae a inglés)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    pass

CL_TZ = pytz.timezone("America/Santiago")

def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Falta variable de entorno GEMINI_API_KEY.")
    return key

# --------------------------------------------------------------------------------------
# Modelos de entrada
# --------------------------------------------------------------------------------------
class MetaFetchInput(BaseModel):
    access_token: str = Field(..., description="Long-lived token de Graph API")
    facebook_page_id: Optional[str] = Field(None)
    instagram_business_id: Optional[str] = Field(None)
    post_limit: int = Field(POST_LIMIT_DEFAULT, ge=1, le=200)
    comment_limit: int = Field(COMMENT_LIMIT_DEFAULT, ge=0, le=500)

class PostItem(BaseModel):
    platform: str
    post_id: str
    created_date: str
    day_of_week: Optional[str] = None
    hour_of_day: Optional[int] = None
    content: Optional[str] = None
    url: Optional[str] = None
    likes_reactions: int = 0
    comments_count: int = 0
    shares: int = 0
    post_comments: str = ""

class RunBody(BaseModel):
    posts: Optional[List[PostItem]] = None
    meta_fetch: Optional[MetaFetchInput] = None
    date_from: Optional[str] = Field(None, description="YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="YYYY-MM-DD")

    @validator("date_from", "date_to")
    def _val_dates(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Formato de fecha inválido. Usa YYYY-MM-DD")
        return v

# --------------------------------------------------------------------------------------
# Helpers de fechas
# --------------------------------------------------------------------------------------
def parse_ts_to_local(ts: str) -> dict:
    try:
        ts = ts.replace("+0000", "+00:00") if ts.endswith("+0000") else ts
        utc_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        local_dt = utc_dt.astimezone(CL_TZ)
        dias_semana = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
        return {
            "fecha_completa": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "dia_semana": dias_semana[local_dt.weekday()],
            "hora_dia": local_dt.hour,
        }
    except Exception:
        base = ts.split("T")[0]
        return {"fecha_completa": base, "dia_semana": "Desconocido", "hora_dia": -1}

def resolve_range(date_from: Optional[str], date_to: Optional[str]):
    if date_from and date_to:
        dfrom = CL_TZ.localize(datetime.strptime(date_from, "%Y-%m-%d"))
        dto = CL_TZ.localize(datetime.strptime(date_to, "%Y-%m-%d")) + timedelta(days=1) - timedelta(seconds=1)
        return dfrom, dto
    # por defecto: mes actual
    now = datetime.now(CL_TZ)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month - timedelta(seconds=1)
    return start, end

# --------------------------------------------------------------------------------------
# Fetchers Meta (filtramos client-side por fecha)
# --------------------------------------------------------------------------------------
def get_facebook_posts(page_id: str, token: str, comment_limit: int, post_limit: int) -> List[dict]:
    fields = (
        f'id,created_time,message,permalink_url,shares,'
        f'reactions.summary(true),comments.limit({comment_limit}).summary(true){{message}}'
    )
    url = f"https://graph.facebook.com/{API_VERSION}/{page_id}/posts"
    params = {"fields": fields, "limit": post_limit, "access_token": token}
    print("Obteniendo publicaciones y comentarios de Facebook...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    posts = r.json().get("data", [])
    out = []
    for p in posts:
        t = parse_ts_to_local(p.get("created_time", ""))
        comments_list = []
        if p.get("comments", {}).get("data"):
            for c in p["comments"]["data"]:
                comments_list.append((c.get("message") or "").replace("\n", " ").replace('"', "'"))
        out.append({
            "platform": "Facebook",
            "post_id": p.get("id"),
            "created_date": t["fecha_completa"],
            "day_of_week": t["dia_semana"],
            "hour_of_day": t["hora_dia"],
            "content": (p.get("message") or "Sin texto").replace("\n"," "),
            "url": p.get("permalink_url"),
            "likes_reactions": p.get("reactions", {}).get("summary", {}).get("total_count", 0),
            "comments_count": p.get("comments", {}).get("summary", {}).get("total_count", 0),
            "shares": p.get("shares", {}).get("count", 0) or 0,
            "post_comments": " ||| ".join(comments_list)
        })
    print(f"-> Facebook: {len(out)} publicaciones")
    return out

def get_instagram_posts(ig_id: str, token: str, comment_limit: int, post_limit: int) -> List[dict]:
    fields = (
        f'id,timestamp,caption,permalink,media_type,like_count,'
        f'comments.limit({comment_limit}).summary(true){{text}}'
    )
    url = f"https://graph.facebook.com/{API_VERSION}/{ig_id}/media"
    params = {"fields": fields, "limit": post_limit, "access_token": token}
    print("Obteniendo publicaciones y comentarios de Instagram...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    posts = r.json().get("data", [])
    out = []
    for p in posts:
        t = parse_ts_to_local(p.get("timestamp", ""))
        comments_list = []
        if p.get("comments", {}).get("data"):
            for c in p["comments"]["data"]:
                comments_list.append((c.get("text") or "").replace("\n"," ").replace('"', "'"))
        out.append({
            "platform": "Instagram",
            "post_id": p.get("id"),
            "created_date": t["fecha_completa"],
            "day_of_week": t["dia_semana"],
            "hour_of_day": t["hora_dia"],
            "content": (p.get("caption") or "Sin texto").replace("\n"," "),
            "url": p.get("permalink"),
            "likes_reactions": p.get("like_count", 0) or 0,
            "comments_count": p.get("comments", {}).get("summary", {}).get("total_count", 0) or 0,
            "shares": 0,
            "post_comments": " ||| ".join(comments_list)
        })
    print(f"-> Instagram: {len(out)} publicaciones")
    return out

# --------------------------------------------------------------------------------------
# IA (Gemini) + pipeline de análisis
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

    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("viridis", len(sentiment_counts)), startangle=90)
    plt.title('Distribución General del Sentimiento', fontsize=16, pad=20)
    plt.savefig('grafico_sentimiento_distribucion.png', dpi=300)

    sentiment_by_platform = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_platform.plot(kind='bar', figsize=(12, 7),
                               color=sns.color_palette("magma", len(sentiment_counts)))
    plt.title('Volumen de Sentimiento por Plataforma', fontsize=16, pad=20)
    plt.xlabel('Plataforma'); plt.ylabel('Cantidad de Publicaciones'); plt.xticks(rotation=0)
    plt.legend(title='Sentimiento'); plt.tight_layout()
    plt.savefig('grafico_sentimiento_por_plataforma.png', dpi=300)

    df['week'] = df['created_date'].dt.strftime('Semana %U')
    sentiment_by_week = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_week.plot(kind='bar', stacked=True, figsize=(12, 7),
                           color=sns.color_palette("plasma", len(sentiment_counts)))
    plt.title('Evolución Semanal del Sentimiento', fontsize=16, pad=20)
    plt.xlabel('Semana del Mes'); plt.ylabel('Cantidad de Publicaciones'); plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentimiento'); plt.tight_layout()
    plt.savefig('grafico_sentimiento_semanal.png', dpi=300)
    plt.close('all')

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.cell(0, 10, 'CONFIDENCIAL | REPORTE ESTRATÉGICO DE REDES SOCIALES', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 10, 'Lider Bci', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(10, 20, 200, 20)
    def footer(self):
        self.set_y(-15); self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')
    def create_title_page(self, title, subtitle, date_range):
        self.add_page(); self.set_y(80); self.set_font('Helvetica', 'B', 28)
        self.multi_cell(0, 15, title, align='C'); self.ln(10); self.set_font('Helvetica', '', 18)
        self.multi_cell(0, 12, subtitle, align='C'); self.ln(20); self.set_font('Helvetica', 'I', 12)
        self.multi_cell(0, 10, f"Período de Análisis: {date_range}", align='C')
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14); self.set_fill_color(230, 230, 230)
        self.cell(0, 10, f' {title}', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT); self.ln(5)
    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        safe = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 7, safe); self.ln()

def generar_pdf_desde_df(df_mes: pd.DataFrame, analysis_period: str, out_path: str) -> str:
    crear_graficos(df_mes)

    total_posts = len(df_mes)
    total_likes = int(df_mes['likes_reactions'].sum())
    total_comments = int(df_mes['comments_count'].sum())
    total_shares = int(df_mes['shares'].sum())
    kpis_generales_str = (
        f"- Total de Publicaciones: {total_posts}\n"
        f"- Total de Likes/Reacciones: {total_likes}\n"
        f"- Total de Comentarios: {total_comments}\n"
        f"- Total de Shares: {total_shares}"
    )

    sentiment_counts = df_mes['sentiment'].value_counts()
    sentimiento_general_str = sentiment_counts.to_string()

    df_mes['week'] = df_mes['created_date'].dt.strftime('Semana %U')
    sentimiento_semanal_str = pd.crosstab(df_mes['week'], df_mes['sentiment']).to_string()
    sentimiento_plataforma_str = pd.crosstab(df_mes['platform'], df_mes['sentiment']).to_string()

    main_prompt = f"""
Rol: Actúa como un Analista Senior de Estrategia Digital para la marca "Lider Bci". Tu objetivo es crear un informe mensual conciso y accionable para la gerencia.
Periodo: {analysis_period}
---
**KPIs Generales:**
{kpis_generales_str}

**Distribución General de Sentimiento (Cantidad de publicaciones):**
{sentimiento_general_str}

**Sentimiento Semanal:**
{sentimiento_semanal_str}

**Sentimiento por Plataforma:**
{sentimiento_plataforma_str}
---
Genera:
1) Conclusión ejecutiva (3–4 líneas).
2) Tabla Markdown con KPIs y distribución de sentimiento.
3) Análisis por semana y por plataforma (breve, accionable).
4) 5 iniciativas concretas para el próximo mes.
""".strip()

    reporte_texto = ia_call_with_retry(main_prompt)

    pdf = PDF('P', 'mm', 'A4')
    pdf.create_title_page('Reporte Estratégico Mensual', 'Inteligencia de Redes Sociales', analysis_period)
    pdf.add_page(); pdf.chapter_title('Análisis y Propuestas Estratégicas'); pdf.chapter_body(reporte_texto)

    pdf.add_page(); pdf.chapter_title('Apéndice Visual')
    for path, title in [
        ('grafico_sentimiento_distribucion.png', 'Distribución General del Sentimiento'),
        ('grafico_sentimiento_por_plataforma.png', 'Volumen de Sentimiento por Plataforma'),
        ('grafico_sentimiento_semanal.png', 'Evolución Semanal del Sentimiento'),
    ]:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, title, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(5)
        pdf.image(path, x=15, w=180)

    pdf.output(out_path)
    print(f"¡Éxito! Reporte guardado como '{out_path}'")
    return out_path

def pipeline_analisis(posts: List[dict], dfrom: datetime, dto: datetime, out_path: str) -> str:
    if not posts:
        raise ValueError("No llegaron posts para analizar.")

    df = pd.DataFrame(posts)

    required = {"platform","post_id","created_date","likes_reactions","comments_count","shares","post_comments"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # 1) parse fechas a datetime
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.dropna(subset=["created_date"])

    # 2) asegurar TZ consistente (America/Santiago)
    #    si la serie es naive -> tz_localize; si ya viene con tz -> tz_convert
    dtype_tz = getattr(df["created_date"].dtype, "tz", None)
    if dtype_tz is None:
        df["created_date"] = df["created_date"].dt.tz_localize(CL_TZ)
    else:
        df["created_date"] = df["created_date"].dt.tz_convert(CL_TZ)

    # 3) filtrar por rango
    df = df[(df["created_date"] >= dfrom) & (df["created_date"] <= dto)].copy()
    if df.empty:
        raise ValueError("No hay publicaciones dentro del rango solicitado.")

    # 4) sentimiento
    df["sentiment"] = df["post_comments"].apply(analizar_sentimiento_comentario)

    # 5) generar PDF
    period_str = f"{dfrom.strftime('%d-%b-%Y')} a {dto.strftime('%d-%b-%Y')}"
    return generar_pdf_desde_df(df, period_str, out_path)

# --------------------------------------------------------------------------------------
# API (FastAPI)
# --------------------------------------------------------------------------------------
api = FastAPI(title="RRSS Reporte API", version="2.0.1")

REPORT_INDEX: Dict[str, str] = {}    # report_id -> pdf path
REPORT_STATUS: Dict[str, str] = {}   # report_id -> "pending" | "ready" | "error"
REPORT_ERRORS: Dict[str, str] = {}   # report_id -> último error legible

@api.get("/health")
def health():
    return {"status": "ok"}

def _background_run(posts: List[dict], dfrom: datetime, dto: datetime, out_path: str, report_id: str):
    try:
        REPORT_STATUS[report_id] = "pending"
        pipeline_analisis(posts, dfrom, dto, out_path)
        REPORT_INDEX[report_id] = out_path
        REPORT_STATUS[report_id] = "ready"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[BG ERROR] {err}")
        log_path = os.path.join(REPORTS_DIR, f"{report_id}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(err + "\n\n")
            f.write(traceback.format_exc())
        REPORT_ERRORS[report_id] = err
        REPORT_STATUS[report_id] = "error"

@api.post("/run")
def run_report(body: RunBody, background_tasks: BackgroundTasks):
    """
    Ejecuta el análisis y genera un PDF.
    Modo A: enviar `posts` (lista de publicaciones ya recolectadas).
    Modo B: enviar `meta_fetch` para que el backend consulte Meta (FB/IG) y use ese material.

    Campos opcionales: `date_from`, `date_to` (YYYY-MM-DD, horario America/Santiago).
    Si no envías fechas, se usa el mes actual.
    """
    dfrom, dto = resolve_range(body.date_from, body.date_to)

    posts: List[dict] = []
    if body.posts:
        posts = [p.dict() for p in body.posts]

    if body.meta_fetch:
        mf = body.meta_fetch
        if not (mf.facebook_page_id or mf.instagram_business_id):
            raise HTTPException(status_code=400, detail="Debes enviar al menos facebook_page_id o instagram_business_id en meta_fetch.")
        try:
            if mf.facebook_page_id:
                fb = get_facebook_posts(mf.facebook_page_id, mf.access_token, mf.comment_limit, mf.post_limit)
                posts.extend(fb)
            if mf.instagram_business_id:
                ig = get_instagram_posts(mf.instagram_business_id, mf.access_token, mf.comment_limit, mf.post_limit)
                posts.extend(ig)
        except requests.exceptions.HTTPError as e:
            try:
                api_json = e.response.json()
            except Exception:
                api_json = {"error": "Meta response non-JSON"}
            raise HTTPException(status_code=e.response.status_code, detail={"meta_error": api_json})

    if not posts:
        raise HTTPException(status_code=400, detail="No se recibieron posts ni se pudo obtener desde Meta.")

    report_id = uuid.uuid4().hex
    outfile = f"Reporte_Estrategico_{report_id}.pdf"
    out_path = os.path.join(REPORTS_DIR, outfile)
    REPORT_STATUS[report_id] = "pending"

    background_tasks.add_task(_background_run, posts, dfrom, dto, out_path, report_id)

    return JSONResponse(
        status_code=202,
        content={
            "report_id": report_id,
            "status_url": f"/status/{report_id}",
            "download_url": f"/download/{report_id}",
            "range_used": {"from": dfrom.strftime("%Y-%m-%d %H:%M:%S%z"), "to": dto.strftime("%Y-%m-%d %H:%M:%S%z")}
        }
    )

@api.get("/status/{report_id}")
def report_status(report_id: str):
    st = REPORT_STATUS.get(report_id)
    if not st:
        raise HTTPException(status_code=404, detail="Reporte no encontrado.")
    payload = {"report_id": report_id, "status": st}
    if st == "ready" and report_id in REPORT_INDEX:
        payload["filename"] = os.path.basename(REPORT_INDEX[report_id])
        payload["download_url"] = f"/download/{report_id}"
    if st == "error" and report_id in REPORT_ERRORS:
        payload["error"] = REPORT_ERRORS[report_id]
        payload["log_url"] = f"/error/{report_id}"
    return JSONResponse(payload)

@api.get("/error/{report_id}")
def report_error_log(report_id: str):
    log_path = os.path.join(REPORTS_DIR, f"{report_id}.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log no encontrado.")
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)

@api.get("/download/{report_id}")
def download_report(report_id: str):
    path = REPORT_INDEX.get(report_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado o expirado.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
