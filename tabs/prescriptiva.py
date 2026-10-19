import streamlit as st
import pandas as pd
import numpy as np

from tabs.limpiar import COLUMNAS_CALIFICACION


SALIDA = "Departure Delay in Minutes"

COLUMNAS_DIGITALES = [
    "Ease of Online booking", "Online boarding", "Online support", "Inflight wifi service"
]
COLUMNAS_COMODIDAD = [
    "Seat comfort", "Leg room service", "Food and drink", "Inflight entertainment"
]

SEG_RETRASO = "Retraso largo en riesgo"
SEG_LEAL = "Cliente leal en riesgo"
SEG_DIGITAL = "No leal con mala experiencia digital"
SEG_COMODIDAD = "Económica con poca comodidad"
SEG_OTROS = "Otros en riesgo"
SEG_MONITOREO = "Monitoreo (riesgo bajo)"

# Regla (en palabras), acción y datos para el plan de decisión de cada segmento
PLAN_POR_SEGMENTO = {
    SEG_RETRASO: {
        "regla": "Riesgo alto y retraso de salida ≥ umbral de retraso largo",
        "accion": "Contacto proactivo de disculpa y seguimiento tras el vuelo",
        "responsable": "Atención a clientes",
        "medicion": "Satisfacción de los contactados vs. un grupo similar sin contactar",
        "riesgo": "Contactar solo después del retraso puede sentirse tarde o insuficiente.",
    },
    SEG_LEAL: {
        "regla": "Riesgo alto y cliente leal (y sin retraso largo)",
        "accion": "Contacto personalizado para conocer su experiencia antes de perderlo",
        "responsable": "Programa de lealtad",
        "medicion": "Retención y satisfacción de los contactados vs. no contactados",
        "riesgo": "Un cliente leal insatisfecho es una señal que hoy no explica el modelo.",
    },
    SEG_DIGITAL: {
        "regla": "Riesgo alto, cliente no leal y calificación digital promedio < 3",
        "accion": "Revisar y mejorar la reserva, el abordaje en línea y el soporte digital",
        "responsable": "Producto digital",
        "medicion": "Calificación digital y satisfacción tras el cambio",
        "riesgo": "Mejorar lo digital no garantiza que el cliente vuelva a elegirnos.",
    },
    SEG_COMODIDAD: {
        "regla": "Riesgo alto, clase Eco / Eco Plus y comodidad promedio < 3",
        "accion": "Investigar asiento, espacio, comida y entretenimiento en esa clase",
        "responsable": "Experiencia a bordo",
        "medicion": "Calificación de comodidad y satisfacción de la clase",
        "riesgo": "Mejorar la comodidad tiene costo; hay que validar que el cliente lo valora.",
    },
    SEG_OTROS: {
        "regla": "Riesgo alto sin ninguna de las reglas anteriores",
        "accion": "Encuesta breve de seguimiento para entender el motivo",
        "responsable": "Investigación de clientes",
        "medicion": "Motivos más frecuentes en las respuestas",
        "riesgo": "Sin hipótesis clara, la acción es exploratoria.",
    },
    SEG_MONITOREO: {
        "regla": "Probabilidad de insatisfacción por debajo del umbral",
        "accion": "Sin acción especial; seguir midiendo satisfacción",
        "responsable": "Analítica",
        "medicion": "Evolución de la probabilidad promedio y de la satisfacción real",
        "riesgo": "Puede haber insatisfechos no detectados por el modelo.",
    },
}

ORDEN_SEGMENTOS = list(PLAN_POR_SEGMENTO.keys())


# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

def asignar_segmentos(df, umbral, retraso_largo):
    """
    Asigna a cada pasajero el PRIMER segmento cuya regla cumple.

    Las reglas se evalúan en el orden de ORDEN_SEGMENTOS. Las calificaciones
    iguales a 0 se ignoran (se interpretan como «no aplica»).
    """
    calificaciones = df[COLUMNAS_CALIFICACION].replace(0, np.nan)
    digital = calificaciones[COLUMNAS_DIGITALES].mean(axis=1)
    comodidad = calificaciones[COLUMNAS_COMODIDAD].mean(axis=1)

    riesgo = df["probabilidad_insatisfaccion"] >= umbral

    condiciones = [
        riesgo & (df[SALIDA] >= retraso_largo),
        riesgo & (df["Customer Type"] == "Loyal Customer"),
        riesgo & (digital < 3),
        riesgo & df["Class"].isin(["Eco", "Eco Plus"]) & (comodidad < 3),
        riesgo,
    ]
    nombres = [SEG_RETRASO, SEG_LEAL, SEG_DIGITAL, SEG_COMODIDAD, SEG_OTROS]

    return pd.Series(
        np.select(condiciones, nombres, default=SEG_MONITOREO),
        index=df.index
    )


def resumen_segmentos(df):
    """Una fila por segmento con pasajeros, probabilidad y % realmente insatisfechos."""
    resumen = (
        df.groupby("segmento")
        .agg(
            pasajeros=("segmento", "size"),
            prob_media=("probabilidad_insatisfaccion", "mean"),
            insatisfechos=("insatisfecho", "mean"),
        )
        .reindex(ORDEN_SEGMENTOS)
        .dropna()
        .reset_index()
    )
    resumen["pct_total"] = resumen["pasajeros"] / len(df) * 100
    resumen["prob_media"] = resumen["prob_media"] * 100
    resumen["insatisfechos"] = resumen["insatisfechos"] * 100
    return resumen


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """
    Muestra la pestaña "Prescriptiva".

    Recibe el dataframe con probabilidades de la pestaña Predictiva y devuelve
    una copia con las columnas `segmento` y `accion_recomendada`
    (o None si todavía no se puede calcular).
    """

    st.header("5. Prescribir: ¿qué acción tomamos con cada tipo de pasajero?")

    st.markdown("""
    ### Pregunta de esta etapa

    **Dado lo que observamos y la probabilidad de insatisfacción, ¿qué acción corresponde a cada grupo?**

    Las recomendaciones salen de **reglas escritas y visibles**, no de una
    caja negra. Cada pasajero recibe la **primera** regla que cumple.
    """)

    if df is None or "probabilidad_insatisfaccion" not in df.columns:
        st.warning(
            "Primero completa la pestaña **Predictiva** (elige al menos un tipo "
            "de información) para calcular las probabilidades."
        )
        return None


    # ----------------------------------
    # PARÁMETROS DE LAS REGLAS
    # ----------------------------------

    st.subheader("Paso 1 · Define qué es «riesgo alto»")

    col1, col2 = st.columns(2)

    with col1:
        umbral_pct = st.slider(
            "Probabilidad mínima de insatisfacción para actuar",
            min_value=30,
            max_value=95,
            value=70,
            step=5,
            format="%d %%",
            key="umbral_prescriptiva",
            help="Más bajo = se contacta a más pasajeros (más costo, menos omisiones)."
        )

    with col2:
        retraso_largo = st.slider(
            "Retraso de salida considerado largo (minutos)",
            min_value=30,
            max_value=180,
            value=60,
            step=15,
            key="retraso_largo_prescriptiva"
        )

    resultado = df.copy()
    resultado["segmento"] = asignar_segmentos(resultado, umbral_pct / 100, retraso_largo)
    resultado["accion_recomendada"] = resultado["segmento"].map(
        {s: p["accion"] for s, p in PLAN_POR_SEGMENTO.items()}
    )

    en_riesgo = int((resultado["segmento"] != SEG_MONITOREO).sum())
    total = len(resultado)

    st.metric(
        "Pasajeros en riesgo alto",
        f"{en_riesgo:,} ({en_riesgo / total * 100:.1f} %)",
        help="Con probabilidad igual o mayor al umbral elegido."
    )


    # ----------------------------------
    # REGLAS Y ACCIONES
    # ----------------------------------

    st.subheader("Paso 2 · Reglas, evidencia y acción por segmento")

    resumen = resumen_segmentos(resultado)

    tabla = pd.DataFrame({
        "Segmento": resumen["segmento"],
        "Regla": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["regla"]),
        "Pasajeros": resumen["pasajeros"],
        "% del total": resumen["pct_total"].round(1),
        "Prob. media de insatisfacción (%)": resumen["prob_media"].round(1),
        "% realmente insatisfechos": resumen["insatisfechos"].round(1),
        "Acción recomendada": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["accion"]),
    })

    st.dataframe(tabla, width="stretch", hide_index=True)

    st.bar_chart(
        tabla.set_index("Segmento")["Pasajeros"],
        horizontal=True,
        sort=False
    )

    st.caption(
        "«% realmente insatisfechos» es lo que respondieron en la encuesta; sirve "
        "para comprobar que las reglas apuntan a grupos con más insatisfacción que "
        "el promedio general "
        f"({resultado['insatisfecho'].mean() * 100:.1f} %)."
    )


    # ----------------------------------
    # ANTES DE DECIDIR
    # ----------------------------------

    st.subheader("Antes de usar estas recomendaciones")

    st.markdown("""
    - 📏 **Las reglas y los umbrales son supuestos**: cámbialos con los controles de arriba y observa cuánto se mueve cada segmento.
    - 🎯 **Una probabilidad no es una garantía.** Habrá pasajeros contactados que no lo necesitaban y otros no contactados que sí.
    - 🔗 **Las acciones son hipótesis.** El dataset muestra asociaciones, no demuestra que cada acción mejore la satisfacción; por eso en la siguiente pestaña se define cómo medirlo.
    - 👀 **Una persona debe revisar** los segmentos antes de ejecutar cualquier acción.
    """)

    return resultado
