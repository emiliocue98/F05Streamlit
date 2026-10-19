import streamlit as st
import pandas as pd


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("1. Entender los datos")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué información tenemos sobre nuestros pasajeros y su nivel de satisfacción, y qué acción deberíamos tomar con cada tipo de pasajero?**
    """)


    # ----------------------------------
    # DIMENSIONES DEL DATASET
    # ----------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Número de pasajeros (encuestas)",
            len(df)
        )

    with col2:
        st.metric(
            "Número de variables",
            len(df.columns)
        )


    # ----------------------------------
    # VISTA PREVIA
    # ----------------------------------

    st.subheader("Vista previa del dataset")

    st.dataframe(
        df.head(10),
        width="stretch"
    )


    # ----------------------------------
    # DIAGNÓSTICO
    # ----------------------------------

    st.subheader("Diagnóstico del dataset")

    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Valores faltantes": df.isnull().sum().values
    })

    st.dataframe(
        diagnostico,
        width="stretch"
    )


    # ----------------------------------
    # DUPLICADOS
    # ----------------------------------

    st.subheader("Problemas potenciales")

    duplicados = df.duplicated().sum()

    st.metric(
        "Registros duplicados",
        duplicados
    )


    # ----------------------------------
    # EXPLORADOR DE VARIABLES
    # ----------------------------------

    st.subheader("Explorar una variable")

    variable = st.selectbox(
        "Selecciona una variable:",
        df.columns,
        key="variable_entender"
    )


    # ----------------------------------
    # VARIABLE NUMÉRICA
    # ----------------------------------

    if pd.api.types.is_numeric_dtype(df[variable]):

        st.write("**Tipo:** Variable numérica")

        estadisticas = df[variable].describe()

        st.dataframe(
            estadisticas,
            width="stretch"
        )


    # ----------------------------------
    # VARIABLE CATEGÓRICA
    # ----------------------------------

    else:

        st.write("**Tipo:** Variable categórica")

        frecuencias = (
            df[variable]
            .value_counts(dropna=False)
            .reset_index()
        )

        frecuencias.columns = [
            "Valor",
            "Frecuencia"
        ]

        st.dataframe(
            frecuencias,
            width="stretch"
        )