import streamlit as st
import pandas as pd


COLUMNA_LLEGADA = "Arrival Delay in Minutes"
COLUMNA_SALIDA = "Departure Delay in Minutes"

# Calificaciones de la encuesta (escala de 1 a 5; el 0 se investiga abajo)
COLUMNAS_CALIFICACION = [
    "Seat comfort", "Departure/Arrival time convenient", "Food and drink",
    "Gate location", "Inflight wifi service", "Inflight entertainment",
    "Online support", "Ease of Online booking", "On-board service",
    "Leg room service", "Baggage handling", "Checkin service",
    "Cleanliness", "Online boarding",
]

OPCION_CONSERVAR = "Conservar como faltante (NaN)"
OPCION_COPIAR_SALIDA = "Reemplazar por el retraso de salida"
OPCION_ELIMINAR = "Eliminar los pasajeros con faltantes"

OPCION_CERO_CONSERVAR = "Conservar el 0 como valor"
OPCION_CERO_FALTANTE = "Convertir el 0 en faltante (NaN)"


# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

def tabla_faltantes(df):
    """Resumen de valores faltantes por variable (solo las que tienen)."""
    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0]

    return pd.DataFrame({
        "Variable": faltantes.index,
        "Faltantes": faltantes.values,
        "% del total": (faltantes.values / len(df) * 100).round(2)
    })


def tabla_ceros_calificaciones(df):
    """Cuántas calificaciones valen 0 en cada pregunta de la encuesta."""
    columnas = [c for c in COLUMNAS_CALIFICACION if c in df.columns]
    ceros = (df[columnas] == 0).sum()

    return pd.DataFrame({
        "Pregunta": columnas,
        "Calificaciones = 0": ceros.values,
        "% del total": (ceros.values / len(df) * 100).round(1)
    }).sort_values("Calificaciones = 0", ascending=False)


def tabla_reglas_validez(df):
    """Cuenta cuántos registros violan cada regla de validez o consistencia."""
    columnas = [c for c in COLUMNAS_CALIFICACION if c in df.columns]

    reglas = [
        ("Edad fuera de 5 a 100 años",
         (df["Age"] < 5) | (df["Age"] > 100)),
        ("Distancia de vuelo menor o igual a 0",
         df["Flight Distance"] <= 0),
        ("Calificaciones fuera de la escala 0 a 5",
         ((df[columnas] < 0) | (df[columnas] > 5)).any(axis=1)),
        ("Retraso de salida negativo",
         df[COLUMNA_SALIDA] < 0),
        ("Retraso de llegada negativo",
         df[COLUMNA_LLEGADA] < 0),
        ("satisfaction distinto de satisfied / dissatisfied",
         ~df["satisfaction"].isin(["satisfied", "dissatisfied"])),
        ("Retraso de llegada 120 min menor que el de salida",
         df[COLUMNA_LLEGADA] < df[COLUMNA_SALIDA] - 120),
    ]

    filas = []
    for nombre, mascara in reglas:
        violaciones = int(mascara.sum())
        filas.append({
            "Regla revisada": nombre,
            "Registros que la violan": violaciones,
            "Resultado": "✅ Sin problemas" if violaciones == 0 else "⚠️ Revisar"
        })

    return pd.DataFrame(filas)


def calcular_limites_iqr(serie):
    """Límites del método IQR (1.5 × rango intercuartil)."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def histograma_simple(serie, intervalos=25):
    """Conteos por intervalo para graficar con st.bar_chart."""
    conteos = pd.cut(serie.dropna(), bins=intervalos).value_counts(sort=False)
    centros = [round(intervalo.mid, 1) for intervalo in conteos.index]
    return pd.DataFrame({"Pasajeros": conteos.values}, index=centros)


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """
    Muestra la pestaña "Limpiar" y devuelve el dataframe limpio.

    El dataframe recibido (df) NO se modifica: es el dato crudo.
    Todas las decisiones se aplican sobre una copia (df_limpio).
    """

    decisiones = []   # bitácora de decisiones para el resumen final

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("2. Limpiar los datos")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Qué problemas tiene el dataset y qué haremos con cada uno?**

    Regla de oro: **primero entendemos, después transformamos.**
    Un dato faltante no es automáticamente un error, y un valor extremo
    no es automáticamente un valor incorrecto.
    Cada tratamiento necesita una **razón escrita**.
    """)

    st.info(
        "Los datos originales no se modifican. "
        "Las decisiones de esta pestaña se aplican a una copia (`df_limpio`), "
        "así que todo es reversible y auditable."
    )


    # ----------------------------------
    # 1. VALORES FALTANTES
    # ----------------------------------

    st.subheader("1. Valores faltantes")

    faltantes = tabla_faltantes(df)

    if faltantes.empty:
        st.success("No hay valores faltantes en el dataset.")
    else:
        st.dataframe(faltantes, width="stretch", hide_index=True)

    st.markdown("#### Investigar antes de decidir")
    st.write(
        "Pregunta: ¿el retraso de llegada falta porque **no hubo retraso** "
        "(sería un 0 implícito) o por otra razón? Si fuera un 0 implícito, "
        "esos pasajeros tendrían también retraso de salida igual a 0."
    )

    falta_llegada = df[COLUMNA_LLEGADA].isnull()
    total_faltantes = int(falta_llegada.sum())
    salida_cero = df[COLUMNA_SALIDA] == 0
    faltantes_salida_cero = int((falta_llegada & salida_cero).sum())
    correlacion = df[COLUMNA_SALIDA].corr(df[COLUMNA_LLEGADA])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pasajeros sin retraso de llegada", total_faltantes)

    with col2:
        st.metric("…de ellos con retraso de salida = 0", faltantes_salida_cero)

    with col3:
        st.metric(
            "Correlación salida vs. llegada",
            f"{correlacion:.2f}",
            help="Calculada solo con los pasajeros que tienen ambos datos."
        )

    cruce = pd.crosstab(
        salida_cero.map({True: "Salida sin retraso (0)", False: "Salida con retraso (>0)"}),
        falta_llegada.map({True: "Falta retraso de llegada", False: "Tiene dato"}),
        rownames=["Retraso de salida"],
        colnames=["Retraso de llegada"]
    )
    st.dataframe(cruce, width="stretch")

    if total_faltantes > 0 and faltantes_salida_cero == total_faltantes:
        st.success(
            "Todos los faltantes tienen retraso de salida = 0: podría ser un "
            "«sin retraso» implícito. Confírmalo con el negocio antes de reemplazar."
        )
    elif total_faltantes > 0:
        st.warning(
            "Los faltantes **no** se explican por la ausencia de retraso: "
            f"{total_faltantes - faltantes_salida_cero} de {total_faltantes} sí "
            "tuvieron retraso de salida. Reemplazarlos por 0 inventaría "
            "información. En cambio, el retraso de salida y el de llegada están "
            "muy relacionados, así que el de salida es una aproximación razonable."
        )

    st.markdown("#### Tu decisión")

    opciones_faltantes = [OPCION_CONSERVAR, OPCION_COPIAR_SALIDA, OPCION_ELIMINAR]

    tratamiento_faltantes = st.radio(
        f"¿Qué hacemos con `{COLUMNA_LLEGADA}` faltante?",
        opciones_faltantes,
        key="tratamiento_faltantes_limpiar"
    )

    if tratamiento_faltantes == OPCION_COPIAR_SALIDA:
        st.warning(
            "Es una **suposición**: un vuelo puede recuperar o perder tiempo en el "
            "aire. Queda documentada en la bitácora."
        )
    elif tratamiento_faltantes == OPCION_ELIMINAR:
        st.warning(
            f"Se eliminarían {total_faltantes} pasajeros "
            f"({total_faltantes / len(df) * 100:.2f} % del total). Es poco, pero "
            "conviene verificar que no sean un grupo particular."
        )
    else:
        st.caption(
            "Se conserva el faltante: es honesto y no inventa información. "
            "Los modelos que se usarán más adelante pueden trabajar con él."
        )

    crear_indicador = st.checkbox(
        "Crear la columna `llegada_faltante` (True si el retraso de llegada estaba vacío)",
        value=False,
        key="indicador_faltante_limpiar"
    )

    razon_faltantes = st.text_area(
        "Razón de tu decisión (obligatoria):",
        key="razon_faltantes_limpiar",
        placeholder="Ejemplo: son muy pocos registros y el retraso de salida "
                    "explica casi todo el de llegada."
    )

    if not razon_faltantes.strip():
        st.caption("⚠️ Aún no escribes la razón de este tratamiento.")

    decisiones.append({
        "Tema": "Faltantes en retraso de llegada",
        "Decisión": tratamiento_faltantes
                    + (" + columna llegada_faltante" if crear_indicador else ""),
        "Razón": razon_faltantes.strip() or "(sin razón escrita)"
    })


    # ----------------------------------
    # 2. DUPLICADOS
    # ----------------------------------

    st.subheader("2. Registros duplicados")

    duplicados_exactos = int(df.duplicated().sum())

    st.metric("Filas totalmente duplicadas", duplicados_exactos)

    st.caption(
        "Este dataset no tiene identificador de pasajero, así que solo se pueden "
        "detectar filas idénticas. Dos pasajeros distintos podrían tener "
        "respuestas iguales por casualidad."
    )

    if duplicados_exactos == 0:
        st.success("No hay duplicados exactos: no se requiere ninguna acción.")
        eliminar_duplicados = False
    else:
        st.warning(
            "Hay filas idénticas. Revisa si son la misma encuesta capturada "
            "dos veces antes de eliminarlas."
        )
        eliminar_duplicados = st.checkbox(
            "Eliminar filas totalmente duplicadas (conservar la primera)",
            value=False,
            key="eliminar_duplicados_limpiar"
        )

    decisiones.append({
        "Tema": "Duplicados",
        "Decisión": "Eliminar filas duplicadas" if eliminar_duplicados
                    else "Sin cambios",
        "Razón": "Se detectaron duplicados exactos." if eliminar_duplicados
                 else "No se encontraron duplicados."
                 if duplicados_exactos == 0
                 else "Se decidió no eliminar tras revisar."
    })


    # ----------------------------------
    # 3. CALIFICACIONES CON VALOR 0
    # ----------------------------------

    st.subheader("3. Calificaciones con valor 0")

    st.write(
        "Las preguntas de la encuesta parecen usar una escala de **1 a 5**. "
        "Entonces, ¿qué significa un **0**? Podría ser «no aplica» "
        "(por ejemplo, el vuelo no ofrecía wifi) o «no respondió», pero no es "
        "una calificación baja. Es una interpretación: valídala con el negocio."
    )

    ceros = tabla_ceros_calificaciones(df)
    st.dataframe(ceros, width="stretch", hide_index=True)

    st.caption(
        "Fíjate que los ceros se concentran en pocas preguntas (asientos, comida, "
        "entretenimiento, horario) y casi no existen en otras: no parecen azar."
    )

    tratamiento_ceros = st.radio(
        "¿Qué hacemos con las calificaciones iguales a 0?",
        [OPCION_CERO_CONSERVAR, OPCION_CERO_FALTANTE],
        key="tratamiento_ceros_limpiar"
    )

    if tratamiento_ceros == OPCION_CERO_FALTANTE:
        st.warning(
            "Al convertirlos en faltantes, promedios y gráficas ya no mezclarán "
            "«no aplica» con «muy mala experiencia». Los faltantes aumentarán."
        )
    else:
        st.caption(
            "Si se conservan, un 0 se promediará como si fuera una calificación "
            "muy baja, lo que puede distorsionar los KPI."
        )

    decisiones.append({
        "Tema": "Calificaciones = 0",
        "Decisión": tratamiento_ceros,
        "Razón": "Se interpretan como «no aplica / sin respuesta» (supuesto a validar)."
                 if tratamiento_ceros == OPCION_CERO_FALTANTE
                 else "Se respeta el dato tal como viene."
    })


    # ----------------------------------
    # 4. RANGOS INVÁLIDOS Y CONSISTENCIA
    # ----------------------------------

    st.subheader("4. Rangos inválidos y consistencia")

    st.write(
        "Reglas de sentido común: un valor puede ser *posible* aunque raro, "
        "pero uno *imposible* es un error."
    )

    reglas = tabla_reglas_validez(df)
    st.dataframe(reglas, width="stretch", hide_index=True)

    total_violaciones = int(reglas["Registros que la violan"].sum())

    if total_violaciones == 0:
        st.success(
            "Ninguna regla se viola. Aun así, estas reglas son supuestos "
            "razonables: valida con el negocio qué significa cada variable."
        )
    else:
        st.warning(
            "Hay registros que violan reglas. Estudia esos casos antes de "
            "corregirlos o eliminarlos; esta pestaña no los modifica sola."
        )

    decisiones.append({
        "Tema": "Rangos y consistencia",
        "Decisión": "Sin cambios",
        "Razón": f"{total_violaciones} violaciones detectadas en las reglas revisadas."
    })


    # ----------------------------------
    # 5. VALORES ATÍPICOS
    # ----------------------------------

    st.subheader("5. Valores atípicos (outliers)")

    st.write(
        "Un valor atípico está lejos del resto, pero **puede ser real** "
        "(por ejemplo, un retraso de más de 24 horas por una tormenta). "
        "No se elimina solo por ser extremo."
    )

    columnas_numericas = ["Age", "Flight Distance", COLUMNA_SALIDA, COLUMNA_LLEGADA]

    variable_atipicos = st.selectbox(
        "Selecciona una variable numérica:",
        columnas_numericas,
        index=columnas_numericas.index(COLUMNA_SALIDA),
        key="variable_atipicos_limpiar"
    )

    serie = df[variable_atipicos].dropna()
    limite_inferior, limite_superior = calcular_limites_iqr(serie)
    es_atipico = (df[variable_atipicos] < limite_inferior) | (df[variable_atipicos] > limite_superior)
    cantidad_atipicos = int(es_atipico.sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Límite inferior (IQR)", f"{limite_inferior:,.1f}")

    with col2:
        st.metric("Límite superior (IQR)", f"{limite_superior:,.1f}")

    with col3:
        st.metric("Valores atípicos", f"{cantidad_atipicos:,}")

    with col4:
        st.metric("% del total", f"{cantidad_atipicos / len(df) * 100:.1f} %")

    st.caption("Distribución de la variable (número de pasajeros por intervalo):")
    st.bar_chart(histograma_simple(serie))

    if cantidad_atipicos > 0:
        st.write("Los 10 valores más extremos (la columna `fila` es la posición en el archivo):")
        extremos = (
            df.loc[es_atipico, [variable_atipicos]]
            .sort_values(variable_atipicos, ascending=False)
            .head(10)
        )
        extremos.index.name = "fila"
        st.dataframe(extremos, width="stretch")

    tratamiento_atipicos = st.radio(
        f"¿Qué hacemos con los atípicos de `{variable_atipicos}`?",
        ["Conservar sin cambios", "Conservar y marcar con una columna indicadora"],
        key="tratamiento_atipicos_limpiar"
    )

    st.caption(
        "Esta pestaña no ofrece eliminar atípicos a propósito: antes de "
        "descartar un pasajero hay que demostrar que el dato es un error."
    )

    marcar_atipicos = tratamiento_atipicos != "Conservar sin cambios"

    decisiones.append({
        "Tema": f"Atípicos en {variable_atipicos}",
        "Decisión": tratamiento_atipicos,
        "Razón": f"{cantidad_atipicos:,} valores fuera de los límites IQR; "
                 "pueden ser eventos reales (retrasos largos, vuelos largos)."
    })


    # ----------------------------------
    # CONSTRUIR EL DATAFRAME LIMPIO
    # ----------------------------------

    df_limpio = df.copy()

    if eliminar_duplicados:
        df_limpio = df_limpio.drop_duplicates()

    if crear_indicador:
        df_limpio["llegada_faltante"] = df_limpio[COLUMNA_LLEGADA].isnull()

    if tratamiento_faltantes == OPCION_COPIAR_SALIDA:
        df_limpio[COLUMNA_LLEGADA] = df_limpio[COLUMNA_LLEGADA].fillna(df_limpio[COLUMNA_SALIDA])
    elif tratamiento_faltantes == OPCION_ELIMINAR:
        df_limpio = df_limpio.dropna(subset=[COLUMNA_LLEGADA])

    if tratamiento_ceros == OPCION_CERO_FALTANTE:
        df_limpio[COLUMNAS_CALIFICACION] = df_limpio[COLUMNAS_CALIFICACION].replace(0, float("nan"))

    if marcar_atipicos:
        df_limpio[f"atipico_{variable_atipicos}"] = (
            (df_limpio[variable_atipicos] < limite_inferior)
            | (df_limpio[variable_atipicos] > limite_superior)
        )


    # ----------------------------------
    # RESUMEN: ANTES VS DESPUÉS
    # ----------------------------------

    st.subheader("6. Resumen: antes vs. después")

    comparacion = pd.DataFrame({
        "Medida": ["Pasajeros (filas)", "Variables (columnas)", "Valores faltantes totales"],
        "Datos originales": [len(df), len(df.columns), int(df.isnull().sum().sum())],
        "Datos limpios": [len(df_limpio), len(df_limpio.columns), int(df_limpio.isnull().sum().sum())]
    })

    st.dataframe(comparacion, width="stretch", hide_index=True)

    st.markdown("#### Bitácora de decisiones")

    st.dataframe(pd.DataFrame(decisiones), width="stretch", hide_index=True)

    if not razon_faltantes.strip():
        st.warning("Falta documentar la razón del tratamiento de faltantes.")

    st.caption(
        "Limpiar no es embellecer los datos: es dejar por escrito qué "
        "cambiamos, por qué y qué perdemos al hacerlo."
    )

    return df_limpio
