import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


COLUMNA_RESULTADO = "insatisfecho"   # 1 = dissatisfied, 0 = satisfied (se crea aquí)
SEMILLA = 42
MUESTRA_IMPORTANCIA = 10000   # filas usadas para medir la importancia (más rápido)

# Información que puede usar el modelo, agrupada en lenguaje de marketing.
GRUPOS_DE_INFORMACION = {
    "Perfil del pasajero": ["Gender", "Customer Type", "Age"],
    "Características del viaje": ["Type of Travel", "Class", "Flight Distance"],
    "Retrasos": ["Departure Delay in Minutes", "Arrival Delay in Minutes"],
    "Servicios digitales (reserva, web, wifi)": [
        "Ease of Online booking", "Online boarding", "Online support",
        "Inflight wifi service", "Checkin service"
    ],
    "Comodidad a bordo": [
        "Seat comfort", "Leg room service", "Cleanliness",
        "Food and drink", "Inflight entertainment", "On-board service"
    ],
    "Servicios en tierra": [
        "Departure/Arrival time convenient", "Gate location", "Baggage handling"
    ],
}

GRUPOS_ACTIVOS_POR_DEFECTO = {grupo: True for grupo in GRUPOS_DE_INFORMACION}

COLUMNAS_TEXTO = ["Gender", "Customer Type", "Type of Travel", "Class"]

NOMBRES_CLAROS = {
    "Gender": "Género",
    "Customer Type": "Tipo de cliente (leal / no leal)",
    "Age": "Edad",
    "Type of Travel": "Motivo del viaje",
    "Class": "Clase",
    "Flight Distance": "Distancia del vuelo",
    "Departure Delay in Minutes": "Retraso de salida (min)",
    "Arrival Delay in Minutes": "Retraso de llegada (min)",
    "Ease of Online booking": "Facilidad de reserva en línea",
    "Online boarding": "Abordaje en línea",
    "Online support": "Soporte en línea",
    "Inflight wifi service": "Wifi a bordo",
    "Checkin service": "Servicio de check-in",
    "Seat comfort": "Comodidad del asiento",
    "Leg room service": "Espacio para las piernas",
    "Cleanliness": "Limpieza",
    "Food and drink": "Comida y bebida",
    "Inflight entertainment": "Entretenimiento a bordo",
    "On-board service": "Servicio a bordo",
    "Departure/Arrival time convenient": "Horario conveniente",
    "Gate location": "Ubicación de la puerta",
    "Baggage handling": "Manejo de equipaje",
}


# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

@st.cache_data(show_spinner=False)
def calcular_probabilidades(df, variables):
    """
    Estima la probabilidad de insatisfacción de cada pasajero.

    Cada pasajero se puntúa con un modelo que NO lo vio al aprender
    (validación cruzada de 5 grupos), para que el resultado sea honesto.
    También calcula qué variables pesan más en la predicción.
    """
    variables = list(variables)
    columnas_texto = [c for c in variables if c in COLUMNAS_TEXTO]
    columnas_numericas = [c for c in variables if c not in COLUMNAS_TEXTO]

    preparacion = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", columnas_numericas),
            ("texto", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columnas_texto),
        ]
    )

    modelo = Pipeline([
        ("preparacion", preparacion),
        ("clasificador", HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            random_state=SEMILLA,
        )),
    ])

    X = df[variables]
    y = df[COLUMNA_RESULTADO].astype(int)

    grupos = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    probabilidades = cross_val_predict(modelo, X, y, cv=grupos, method="predict_proba")[:, 1]

    modelo.fit(X, y)
    muestra = X.sample(min(MUESTRA_IMPORTANCIA, len(X)), random_state=SEMILLA)
    importancia = permutation_importance(
        modelo, muestra, y.loc[muestra.index], scoring="roc_auc",
        n_repeats=3, random_state=SEMILLA
    )

    return {
        "probabilidades": pd.Series(probabilidades, index=df.index),
        "importancia": pd.Series(importancia.importances_mean, index=variables),
        "puntaje_general": float(roc_auc_score(y, probabilidades)),
    }


def curva_de_ganancia(resultado_ordenado):
    """Porcentaje de insatisfechos alcanzados según el % de pasajeros contactados."""
    total_pasajeros = len(resultado_ordenado)
    total_insatisfechos = resultado_ordenado.sum()
    acumulado = resultado_ordenado.cumsum().to_numpy()

    filas = []
    for porcentaje in range(0, 101, 5):
        contactados = round(total_pasajeros * porcentaje / 100)
        alcanzados = acumulado[contactados - 1] if contactados > 0 else 0
        filas.append({
            "% de pasajeros contactados": porcentaje,
            "Con el modelo": alcanzados / total_insatisfechos * 100,
            "Al azar": float(porcentaje),
        })

    return pd.DataFrame(filas).set_index("% de pasajeros contactados")


def describir_diferencia(df, variable):
    """Frase corta: qué distingue a los pasajeros insatisfechos en esta variable."""
    insatisfecho = df[COLUMNA_RESULTADO] == 1

    if variable in COLUMNAS_TEXTO:
        tasas = df.groupby(variable)[COLUMNA_RESULTADO].agg(["mean", "size"])
        tasas = tasas[tasas["size"] >= 30]
        if tasas.empty:
            return "—"
        peor = tasas["mean"].idxmax()
        return f"Más insatisfacción en «{peor}» ({tasas.loc[peor, 'mean'] * 100:.0f} %)"

    promedio_insatisfechos = df.loc[insatisfecho, variable].mean()
    promedio_resto = df.loc[~insatisfecho, variable].mean()
    return f"Insatisfechos: {promedio_insatisfechos:,.1f} · Satisfechos: {promedio_resto:,.1f}"


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """
    Muestra la pestaña "Predictiva".

    Recibe el dataframe limpio y devuelve una copia con las columnas
    `insatisfecho` y `probabilidad_insatisfaccion` (o None si todavía
    no se puede calcular).
    """

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("4. Predecir: ¿qué pasajeros tienen más probabilidad de estar insatisfechos?")

    st.markdown("""
    ### Pregunta de esta etapa

    **Si queremos recuperar la experiencia de los pasajeros, ¿a quiénes conviene contactar primero?**

    Un modelo de aprendizaje automático revisa la **encuesta de satisfacción**
    (quién quedó satisfecho y quién no) y aprende qué características se repiten
    en quienes quedaron **insatisfechos**. Con eso asigna a cada pasajero una
    **probabilidad de insatisfacción**.
    """)

    st.info(
        "El modelo no adivina el futuro: ordena a los pasajeros de **más a menos "
        "probable** de estar insatisfechos para ayudarte a decidir dónde invertir "
        "tu presupuesto de retención."
    )

    if "satisfaction" not in df.columns or df["satisfaction"].nunique() < 2:
        st.error("No hay datos suficientes de satisfacción para entrenar el modelo.")
        return None

    pasajeros = df.copy()
    pasajeros[COLUMNA_RESULTADO] = (pasajeros["satisfaction"] == "dissatisfied").astype(int)


    # ----------------------------------
    # PASO 1: QUÉ INFORMACIÓN USAR
    # ----------------------------------

    st.subheader("Paso 1 · ¿Qué información puede usar el modelo?")

    st.write(
        "Marca los tipos de información que el modelo podrá consultar. "
        "Prueba distintas combinaciones y observa cómo cambian los resultados."
    )

    variables = []
    columnas_pantalla = st.columns(2)

    for posicion, (grupo, columnas_grupo) in enumerate(GRUPOS_DE_INFORMACION.items()):
        with columnas_pantalla[posicion % 2]:
            usar_grupo = st.checkbox(
                grupo,
                value=GRUPOS_ACTIVOS_POR_DEFECTO[grupo],
                key=f"grupo_{posicion}_predictiva",
                help="Incluye: " + ", ".join(NOMBRES_CLAROS[c] for c in columnas_grupo)
            )
        if usar_grupo:
            variables.extend(c for c in columnas_grupo if c in df.columns)

    st.caption(
        "⚠️ Las calificaciones de **servicios** se recogen en la **misma encuesta** "
        "que la satisfacción. Sirven para **explicar qué pesa más**, pero no las "
        "tendrías antes de que el pasajero responda. Apágalas para ver qué tan bien "
        "se predice solo con perfil, viaje y retrasos."
    )

    if not variables:
        st.warning("Selecciona al menos un tipo de información para continuar.")
        return None


    # ----------------------------------
    # CÁLCULO
    # ----------------------------------

    with st.spinner("El modelo está aprendiendo de tus pasajeros (puede tardar unos segundos)..."):
        resultado = calcular_probabilidades(
            pasajeros[variables + [COLUMNA_RESULTADO]], tuple(variables)
        )

    pasajeros["probabilidad_insatisfaccion"] = resultado["probabilidades"]

    ordenados = pasajeros.sort_values("probabilidad_insatisfaccion", ascending=False)
    insatisfecho_ordenado = ordenados[COLUMNA_RESULTADO].astype(int)

    total_pasajeros = len(pasajeros)
    total_insatisfechos = int(insatisfecho_ordenado.sum())
    tasa_general = total_insatisfechos / total_pasajeros * 100


    # ----------------------------------
    # PASO 2: ¿QUÉ TAN BIEN FUNCIONA?
    # ----------------------------------

    st.subheader("Paso 2 · ¿Qué tan útil es el modelo?")

    st.write(
        f"En la encuesta, **{total_insatisfechos:,} de {total_pasajeros:,} "
        f"pasajeros ({tasa_general:.1f} %)** quedaron insatisfechos. Si contactas "
        "pasajeros al azar, ese es el resultado que esperarías. ¿Mejora el modelo esa situación?"
    )

    porcentaje_contactar = st.slider(
        "¿A qué porcentaje de tus pasajeros enviarías la acción de recuperación?",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        format="%d %%",
        key="porcentaje_contactar_predictiva",
        help="El modelo elige primero a los pasajeros con mayor probabilidad de insatisfacción."
    )

    cantidad_contactar = max(1, round(total_pasajeros * porcentaje_contactar / 100))
    alcanzados = int(insatisfecho_ordenado.iloc[:cantidad_contactar].sum())
    porcentaje_alcanzado = alcanzados / total_insatisfechos * 100
    tasa_contactados = alcanzados / cantidad_contactar * 100
    veces_mejor = tasa_contactados / tasa_general if tasa_general > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pasajeros a contactar", f"{cantidad_contactar:,}")

    with col2:
        st.metric(
            "Insatisfechos que alcanzarías",
            f"{porcentaje_alcanzado:.0f} %",
            help="Del total de pasajeros que quedaron insatisfechos en la encuesta."
        )

    with col3:
        st.metric(
            "Rinde vs. contactar al azar",
            f"{veces_mejor:.1f} veces",
            help="Cuántas veces más insatisfechos por cada pasajero contactado."
        )

    mensaje = (
        f"Si contactas al **{porcentaje_contactar} %** de tus pasajeros "
        f"({cantidad_contactar:,}) elegidos por el modelo, habrías llegado a "
        f"**{porcentaje_alcanzado:.0f} %** de los insatisfechos. "
        f"Al azar solo habrías llegado a {porcentaje_contactar} %."
    )

    if porcentaje_contactar < 100 and veces_mejor >= 1.5:
        st.success("✅ El modelo ayuda mucho a priorizar. " + mensaje)
    elif porcentaje_contactar < 100 and veces_mejor >= 1.2:
        st.info("🙂 El modelo ayuda de forma moderada. " + mensaje)
    elif porcentaje_contactar < 100:
        st.warning("⚠️ El modelo casi no mejora el azar con esta información. " + mensaje)
    else:
        st.caption("Contactar al 100 % equivale a no priorizar. Baja el porcentaje para ver la ventaja del modelo.")

    st.markdown("**¿Cuántos insatisfechos alcanzas según a cuántos pasajeros contactes?**")
    st.line_chart(
        curva_de_ganancia(insatisfecho_ordenado),
        x_label="% de pasajeros contactados",
        y_label="% de insatisfechos alcanzados"
    )
    st.caption(
        "Mientras más separada esté la línea «Con el modelo» de la línea «Al azar», "
        "más útil es el modelo."
    )

    with st.expander("Ver un detalle más"):
        st.write(
            f"Puntaje general del modelo: **{resultado['puntaje_general']:.2f}** "
            "(0.50 equivale a adivinar y 1.00 sería perfecto)."
        )
        st.write(
            f"Entre los {cantidad_contactar:,} pasajeros contactados: "
            f"**{alcanzados:,}** sí estaban insatisfechos y "
            f"**{cantidad_contactar - alcanzados:,}** no. "
            f"Además, **{total_insatisfechos - alcanzados:,}** insatisfechos "
            "quedaron fuera de la lista. El modelo se equivoca en ambos sentidos."
        )
        st.write(
            "Para que la evaluación sea justa, a cada pasajero lo puntuó un modelo "
            "que no lo había visto antes durante su aprendizaje."
        )


    # ----------------------------------
    # PASO 3: QUÉ FACTORES PESAN MÁS
    # ----------------------------------

    st.subheader("Paso 3 · ¿Qué señales pesan más?")

    st.write(
        "Estas son las características que más ayudaron al modelo a distinguir "
        "a los pasajeros insatisfechos de los satisfechos."
    )

    importancia = resultado["importancia"].clip(lower=0)

    if importancia.sum() > 0:
        peso = (importancia / importancia.sum() * 100).sort_values(ascending=False).head(8)

        tabla_senales = pd.DataFrame({
            "Señal": [NOMBRES_CLAROS[v] for v in peso.index],
            "Importancia (%)": peso.round(0).astype(int).values,
            "Qué se observa": [describir_diferencia(pasajeros, v) for v in peso.index],
        })

        st.bar_chart(
            tabla_senales.set_index("Señal")["Importancia (%)"],
            horizontal=True,
            sort=False
        )
        st.dataframe(tabla_senales, width="stretch", hide_index=True)
    else:
        st.info("Con esta información el modelo no encontró señales claras.")

    st.caption(
        "Una señal **acompaña** a la insatisfacción, pero no demuestra que la "
        "**cause**. Por ejemplo, que quienes califican mal el entretenimiento "
        "estén más insatisfechos no prueba que mejorarlo, por sí solo, los vaya a satisfacer."
    )


    # ----------------------------------
    # PASO 4: LISTA DE PASAJEROS
    # ----------------------------------

    st.subheader("Paso 4 · Lista de pasajeros por prioridad")

    st.write(
        f"Estos son los pasajeros ordenados de mayor a menor probabilidad de "
        f"insatisfacción. Los primeros {cantidad_contactar:,} serían los elegidos."
    )

    lista = ordenados.copy()
    lista.index.name = "fila"
    lista = lista.reset_index()
    lista["Prioridad"] = ["Contactar"] * cantidad_contactar + ["No contactar"] * (total_pasajeros - cantidad_contactar)
    lista["Probabilidad de insatisfacción"] = (lista["probabilidad_insatisfaccion"] * 100).round(1)

    lista = lista[[
        "fila", "Probabilidad de insatisfacción", "Prioridad",
        "Customer Type", "Class", "Type of Travel", "Age"
    ]].rename(columns={
        "fila": "Fila en el archivo",
        "Customer Type": "Tipo de cliente",
        "Class": "Clase",
        "Type of Travel": "Motivo del viaje",
        "Age": "Edad",
    })

    st.dataframe(
        lista.head(100),
        width="stretch",
        hide_index=True,
        column_config={
            "Probabilidad de insatisfacción": st.column_config.ProgressColumn(
                "Probabilidad de insatisfacción",
                min_value=0,
                max_value=100,
                format="%.1f %%"
            )
        }
    )
    st.caption("Se muestran los 100 pasajeros con mayor probabilidad. La descarga incluye a todos.")

    st.download_button(
        "⬇️ Descargar lista completa (CSV)",
        data=lista.to_csv(index=False).encode("utf-8-sig"),
        file_name="pasajeros_priorizados.csv",
        mime="text/csv",
        key="descargar_lista_predictiva"
    )


    # ----------------------------------
    # ANTES DE DECIDIR
    # ----------------------------------

    st.subheader("Antes de usar estos resultados")

    st.markdown("""
    - 🧪 **Es un dataset de práctica.** No hay identificador de pasajero ni fechas, y no sabemos cómo se recogió la encuesta. Sirve para aprender el método, no para decisiones reales.
    - 🎯 **Es una probabilidad, no una certeza.** Algunos pasajeros con probabilidad alta estarán satisfechos y algunos con probabilidad baja no.
    - 🔗 **Acompañar no es causar.** El modelo detecta patrones; no explica por qué alguien queda insatisfecho.
    - 📋 **La satisfacción se mide en la misma encuesta que las calificaciones de servicio.** Úsalas para entender causas, no como si las conocieras antes del vuelo.
    - 👀 **Una persona debe revisar la lista** antes de actuar: ¿tiene sentido?, ¿estamos dejando fuera a algún grupo importante (por clase, tipo de cliente)?
    """)

    return pasajeros
