
from pathlib import Path

import pandas as pd
import streamlit as st

from tabs.entender import mostrar as mostrar_entender
from tabs.limpiar import mostrar as mostrar_limpiar
from tabs.descriptiva import mostrar as mostrar_descriptiva
from tabs.predictiva import mostrar as mostrar_predictiva
from tabs.prescriptiva import mostrar as mostrar_prescriptiva
from tabs.decisiones import mostrar as mostrar_decisiones


RUTA_DATASET = (
    Path(__file__).resolve().parent
    / "data"
    / "retencion_alumnos.csv"
)


st.set_page_config(
    page_title="Retención de Alumnos",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background: #f5f7fb; }
      [data-testid="stHeader"] { background: rgba(245, 247, 251, .85); }
      .main .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #14213d; letter-spacing: -.02em; }
      [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e4eaf2; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 4px 14px rgba(20, 33, 61, .04);
      }
      [data-testid="stMetricLabel"] { color: #52627a; }
      [data-testid="stMetricValue"] { color: #14213d; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 8px; }
      [data-testid="stTabs"] button[role="tab"] {
        background: #fff; border: 1px solid #e4eaf2; border-radius: 10px 10px 0 0;
        padding: 10px 16px; color: #52627a;
      }
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #155eef; border-bottom: 3px solid #155eef; font-weight: 700;
      }
      [data-testid="stDataFrame"] { border: 1px solid #e4eaf2; border-radius: 12px; }
      [data-testid="stDownloadButton"] button, .stButton button {
        border-radius: 10px; font-weight: 600;
      }
      [data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sistema de Decisión con Datos: Retención de Alumnos")
st.markdown(
    "Aplicación didáctica para identificar qué alumnos tienen mayor riesgo de "
    "abandonar y convertir la evidencia en acciones de acompañamiento académico. "
    "Los datos son sintéticos y están anonimizados."
)


@st.cache_data
def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Carga el CSV original sin modificar ni imputar sus datos."""
    return pd.read_csv(ruta)


if not RUTA_DATASET.is_file():
    st.error(
        "No se encontró el dataset esperado: "
        f"`{RUTA_DATASET.relative_to(Path(__file__).resolve().parent)}`. "
        "Verifica que el archivo esté dentro de la carpeta `data`."
    )
    st.stop()

try:
    df = cargar_dataset(RUTA_DATASET)
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"No se pudo leer el dataset: {error}")
    st.stop()

if df.empty:
    st.error("El dataset está vacío. Revisa el archivo CSV antes de continuar.")
    st.stop()


etiquetas_pestanas = [
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones",
]
tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs(
    etiquetas_pestanas
)

with tab_entender:
    mostrar_entender(df)

with tab_limpiar:
    mostrar_limpiar(df)

df_analisis = st.session_state.get("df_limpio", df)

with tab_descriptiva:
    mostrar_descriptiva(df_analisis)

with tab_predictiva:
    mostrar_predictiva(df_analisis)

with tab_prescriptiva:
    mostrar_prescriptiva(df_analisis)

with tab_decisiones:
    mostrar_decisiones(df_analisis)