import streamlit as st
import pandas as pd

from tabs.limpiar import COLUMNAS_CALIFICACION
from tabs.predictiva import NOMBRES_CLAROS


SALIDA = "Departure Delay in Minutes"
UMBRAL_RETRASO = 15   # minutos; práctica común para hablar de "retraso"

# Variables por las que se puede segmentar la satisfacción
OPCIONES_SEGMENTO = {
    "Tipo de cliente": "Customer Type",
    "Motivo del viaje": "Type of Travel",
    "Clase": "Class",
    "Género": "Gender",
    "Grupo de edad": "grupo_edad",
    "Retraso de salida": "grupo_retraso",
}


# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

def agregar_grupos(df):
    """Devuelve una copia con columnas auxiliares para segmentar."""
    df = df.copy()
    df["satisfecho"] = (df["satisfaction"] == "satisfied").astype(int)
    df["grupo_edad"] = pd.cut(
        df["Age"],
        bins=[0, 24, 39, 54, 69, 120],
        labels=["Hasta 24", "25 a 39", "40 a 54", "55 a 69", "70 o más"]
    )
    df["grupo_retraso"] = pd.cut(
        df[SALIDA],
        bins=[-1, 0, UMBRAL_RETRASO, 60, 100000],
        labels=["Sin retraso", "1 a 15 min", "16 a 60 min", "Más de 60 min"]
    )
    return df


def tabla_por_segmento(df, columna):
    """Pasajeros y % de satisfechos por cada valor de una variable."""
    tabla = (
        df.groupby(columna, observed=True)["satisfecho"]
        .agg(["size", "mean"])
        .reset_index()
    )
    tabla.columns = ["Segmento", "Pasajeros", "% satisfechos"]
    tabla["% satisfechos"] = (tabla["% satisfechos"] * 100).round(1)
    tabla["% del total"] = (tabla["Pasajeros"] / len(df) * 100).round(1)
    return tabla


def tabla_brecha_servicios(df):
    """Calificación promedio por servicio, satisfechos vs. insatisfechos (sin los 0)."""
    columnas = [c for c in COLUMNAS_CALIFICACION if c in df.columns]
    calificaciones = df[columnas].replace(0, float("nan"))
    promedios = calificaciones.groupby(df["satisfecho"]).mean().T

    tabla = pd.DataFrame({
        "Servicio": [NOMBRES_CLAROS[c] for c in promedios.index],
        "Satisfechos": promedios[1].round(2).values,
        "Insatisfechos": promedios[0].round(2).values,
    })
    tabla["Brecha"] = (tabla["Satisfechos"] - tabla["Insatisfechos"]).round(2)
    return tabla.sort_values("Brecha", ascending=False).reset_index(drop=True)


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """Muestra la pestaña "Descriptiva" con los KPI de satisfacción."""

    st.header("3. Estadística descriptiva")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Cómo está la satisfacción hoy y en qué tipos de pasajeros es más alta o más baja?**

    Aquí **describimos** lo que ya pasó. Todavía no explicamos causas ni decidimos acciones.
    """)

    datos = agregar_grupos(df)
    total = len(datos)


    # ----------------------------------
    # KPI GENERALES
    # ----------------------------------

    st.subheader("Indicadores generales (KPI)")

    satisfechos = int(datos["satisfecho"].sum())
    pct_satisfechos = satisfechos / total * 100
    pct_leales = (datos["Customer Type"] == "Loyal Customer").mean() * 100
    pct_negocios = (datos["Type of Travel"] == "Business travel").mean() * 100
    pct_retraso = (datos[SALIDA] > UMBRAL_RETRASO).mean() * 100
    calif_media = datos[COLUMNAS_CALIFICACION].replace(0, float("nan")).stack().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pasajeros", f"{total:,}")
    col2.metric("% satisfechos", f"{pct_satisfechos:.1f} %",
                help="Satisfechos / total de pasajeros encuestados.")
    col3.metric("% insatisfechos", f"{100 - pct_satisfechos:.1f} %")
    col4.metric("Edad promedio", f"{datos['Age'].mean():.1f} años")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("% clientes leales", f"{pct_leales:.1f} %")
    col6.metric("% viajes de negocios", f"{pct_negocios:.1f} %")
    col7.metric(f"% con salida retrasada (>{UMBRAL_RETRASO} min)", f"{pct_retraso:.1f} %")
    col8.metric("Calificación media de servicios", f"{calif_media:.2f} / 5",
                help="Promedio de las 14 preguntas. Los 0 se excluyen: se interpretan como «no aplica».")

    st.caption(
        "Todos los porcentajes usan como denominador el **total de pasajeros** "
        "(o el grupo indicado). Las calificaciones excluyen los 0."
    )


    # ----------------------------------
    # SATISFACCIÓN POR SEGMENTO
    # ----------------------------------

    st.subheader("Satisfacción por tipo de pasajero")

    nombre_segmento = st.selectbox(
        "Compara la satisfacción según:",
        list(OPCIONES_SEGMENTO.keys()),
        key="segmento_descriptiva"
    )

    tabla = tabla_por_segmento(datos, OPCIONES_SEGMENTO[nombre_segmento])
    tabla["Diferencia vs. total (pts)"] = (tabla["% satisfechos"] - pct_satisfechos).round(1)

    col_tabla, col_grafica = st.columns(2)

    with col_tabla:
        st.dataframe(tabla, width="stretch", hide_index=True)

    with col_grafica:
        st.bar_chart(tabla.set_index("Segmento")["% satisfechos"])

    mejor = tabla.loc[tabla["% satisfechos"].idxmax()]
    peor = tabla.loc[tabla["% satisfechos"].idxmin()]

    st.write(
        f"Dentro de **{nombre_segmento.lower()}**, el grupo con mayor satisfacción es "
        f"**{mejor['Segmento']}** ({mejor['% satisfechos']:.1f} %) y el menor es "
        f"**{peor['Segmento']}** ({peor['% satisfechos']:.1f} %), una diferencia de "
        f"{mejor['% satisfechos'] - peor['% satisfechos']:.1f} puntos porcentuales."
    )

    st.caption(
        "Es una diferencia **descriptiva**: los grupos también difieren en otras cosas "
        "(por ejemplo, la clase Business suele ser de viajes de negocios), así que "
        "no sabemos todavía cuál factor explica la diferencia."
    )


    # ----------------------------------
    # CRUCE DE DOS VARIABLES
    # ----------------------------------

    st.subheader("Cruce de dos variables")

    cruce = (
        datos.groupby(["Class", "Type of Travel"], observed=True)["satisfecho"]
        .agg(["size", "mean"])
        .reset_index()
    )
    cruce.columns = ["Clase", "Motivo del viaje", "Pasajeros", "% satisfechos"]
    cruce["% satisfechos"] = (cruce["% satisfechos"] * 100).round(1)

    matriz = cruce.pivot(index="Clase", columns="Motivo del viaje", values="% satisfechos")
    st.write("**% de satisfechos** por clase y motivo del viaje:")
    st.dataframe(matriz, width="stretch")

    st.dataframe(cruce, width="stretch", hide_index=True)


    # ----------------------------------
    # SERVICIOS
    # ----------------------------------

    st.subheader("Calificación de servicios: satisfechos vs. insatisfechos")

    st.write(
        "Calificación promedio (1 a 5) de cada servicio entre quienes quedaron "
        "satisfechos y quienes no. La **brecha** indica dónde se separan más."
    )

    brecha = tabla_brecha_servicios(datos)

    col_tabla, col_grafica = st.columns(2)

    with col_tabla:
        st.dataframe(brecha, width="stretch", hide_index=True)

    with col_grafica:
        st.bar_chart(brecha.set_index("Servicio")["Brecha"], horizontal=True, sort=False)

    st.caption(
        "Una brecha grande **acompaña** a la insatisfacción pero no prueba que "
        "mejorar ese servicio cause satisfacción. Las calificaciones y la "
        "satisfacción se registran en la misma encuesta."
    )


    # ----------------------------------
    # ANTES DE PASAR A LA SIGUIENTE ETAPA
    # ----------------------------------

    st.subheader("Lo que ya sabemos")

    st.markdown(f"""
    - De **{total:,}** pasajeros, **{pct_satisfechos:.1f} %** quedaron satisfechos.
    - La satisfacción cambia según el tipo de cliente, clase y motivo del viaje: ahí hay oportunidades de segmentar.
    - Los servicios con mayor brecha son los primeros candidatos a investigar: **{brecha.loc[0, 'Servicio']}**, **{brecha.loc[1, 'Servicio']}** y **{brecha.loc[2, 'Servicio']}**.
    """)
