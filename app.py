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
from fastapi.middleware.cors import CORSMiddleware
from zipfile import ZipFile, ZIP_DEFLATED


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

from fastapi.middleware.cors import CORSMiddleware

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # todos los orígenes
    allow_credentials=False, # debe ser False si usas "*"
    allow_methods=["*"],    # GET, POST, OPTIONS, etc.
    allow_headers=["*"],    # todos los headers
    expose_headers=["*"],   # opcional: exponer headers al cliente
    max_age=600,            # cache del preflight
)

REPORT_INDEX: Dict[str, str] = {}    # report_id -> pdf path
REPORT_STATUS: Dict[str, str] = {}   # report_id -> "pending" | "ready" | "error"
REPORT_ERRORS: Dict[str, str] = {}   # report_id -> último error legible

@api.get("/health")
def health():
    return {"status": "ok"}

def _background_run(posts: List[dict], dfrom: datetime, dto: datetime, out_path: str, report_id: str):
    """
    Ejecuta el pipeline en segundo plano, guarda los datos crudos (JSON/CSV)
    y genera el PDF. Actualiza los estados en REPORT_STATUS / REPORT_INDEX.
    """
    try:
        REPORT_STATUS[report_id] = "pending"

        # --- GUARDAR DATOS CRUDOS ---
        raw_base = os.path.join(REPORTS_DIR, f"raw_posts_{report_id}")
        raw_json = f"{raw_base}.json"
        raw_csv = f"{raw_base}.csv"

        try:
            import json
            with open(raw_json, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            pd.DataFrame(posts).to_csv(raw_csv, index=False, encoding="utf-8")
            print(f"[OK] RAWs guardados en {raw_json} / {raw_csv}")
        except Exception as e:
            print(f"[WARN] No se pudieron escribir los RAWs: {e}")

        # --- GENERAR PDF ---
        pipeline_analisis(posts, dfrom, dto, out_path)

        REPORT_INDEX[report_id] = out_path
        REPORT_STATUS[report_id] = "ready"
        print(f"[OK] Reporte {report_id} listo: {out_path}")

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
    """
    Devuelve un .zip armado al vuelo con:
    - Reporte_Estrategico.pdf
    - datos/raw_posts.json (si existe)
    - datos/raw_posts.csv (si existe)
    - graficos/*.png (si existen)
    """
    pdf_path = REPORT_INDEX.get(report_id)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado o expirado.")

    # --- Nombre del ZIP ---
    zip_name = f"Reporte_{report_id}.zip"
    zip_path = os.path.join(REPORTS_DIR, zip_name)

    # --- Crear ZIP ---
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        # PDF principal
        z.write(pdf_path, arcname="Reporte_Estrategico.pdf")

        # Datos crudos (si existen)
        base_raw = os.path.join(REPORTS_DIR, f"raw_posts_{report_id}")
        raw_json = f"{base_raw}.json"
        raw_csv = f"{base_raw}.csv"
        if os.path.exists(raw_json):
            z.write(raw_json, arcname="datos/raw_posts.json")
        if os.path.exists(raw_csv):
            z.write(raw_csv, arcname="datos/raw_posts.csv")

        # Gráficos (si existen)
        for fn, arc in [
            ("grafico_sentimiento_distribucion.png", "graficos/sentimiento_distribucion.png"),
            ("grafico_sentimiento_por_plataforma.png", "graficos/sentimiento_por_plataforma.png"),
            ("grafico_sentimiento_semanal.png", "graficos/sentimiento_semanal.png"),
        ]:
            p = os.path.join(os.getcwd(), fn)
            if os.path.exists(p):
                z.write(p, arcname=arc)

    print(f"[OK] ZIP generado: {zip_path}")
    return FileResponse(zip_path, media_type="application/zip", filename=zip_name)


