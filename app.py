import streamlit as st
import pandas as pd
from pathlib import Path

from tabs.entender import mostrar as mostrar_entender
from tabs.limpiar import mostrar as mostrar_limpiar
from tabs.descriptiva import mostrar as mostrar_descriptiva
from tabs.predictiva import mostrar as mostrar_predictiva
from tabs.prescriptiva import mostrar as mostrar_prescriptiva
from tabs.decisiones import mostrar as mostrar_decisiones


# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------

st.set_page_config(
    page_title="Invistico Airline Passengers",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------
# TÍTULO DE LA APLICACIÓN
# ----------------------------------

st.title("📊 Invistico Airline Passengers")

st.markdown("""
Aplicación para analizar la satisfacción de los pasajeros de una aerolínea
y convertir datos en decisiones de marketing.
""")


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

RUTA_DATOS = Path(__file__).parent / "data" / "Invistico_Airline.csv"


@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


df = cargar_datos()


# ----------------------------------
# CREAR PESTAÑAS
# ----------------------------------

tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs([
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones"
])


# ----------------------------------
# 1. ENTENDER LOS DATOS
# ----------------------------------

with tab_entender:
    mostrar_entender(df)


# ----------------------------------
# 2. LIMPIAR LOS DATOS
# ----------------------------------

with tab_limpiar:
    df_limpio = mostrar_limpiar(df)


# ----------------------------------
# 3. DESCRIBIR
# ----------------------------------

with tab_descriptiva:
    mostrar_descriptiva(df_limpio)


# ----------------------------------
# 4. PREDECIR
# ----------------------------------

with tab_predictiva:
    df_predicho = mostrar_predictiva(df_limpio)


# ----------------------------------
# 5. PRESCRIBIR
# ----------------------------------

with tab_prescriptiva:
    df_prescrito = mostrar_prescriptiva(df_predicho)


# ----------------------------------
# 6. DECIDIR
# ----------------------------------

with tab_decisiones:
    mostrar_decisiones(df_prescrito)