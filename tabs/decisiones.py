import streamlit as st
import pandas as pd

from tabs.prescriptiva import (
    ORDEN_SEGMENTOS, PLAN_POR_SEGMENTO, SEG_MONITOREO, resumen_segmentos
)


NOMBRES_CLASE = {"Business": "Business", "Eco": "Económica", "Eco Plus": "Económica Plus"}
NOMBRES_VIAJE = {"Business travel": "de negocios", "Personal Travel": "personal"}


# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

def mensaje_ejemplo(segmento, clase, viaje):
    """
    Mensaje de ejemplo basado solo en datos que conocemos (clase y motivo del viaje).
    No promete descuentos, compensaciones ni beneficios que el dataset no menciona.
    """
    contexto = f"en clase {NOMBRES_CLASE.get(clase, clase)}, viaje {NOMBRES_VIAJE.get(viaje, viaje)}"

    mensajes = {
        "Retraso largo en riesgo":
            f"Sabemos que tu vuelo ({contexto}) salió con retraso y lamentamos las molestias. "
            "Nos gustaría escuchar cómo fue tu experiencia para mejorar.",
        "Cliente leal en riesgo":
            f"Gracias por volar con nosotros de forma constante ({contexto}). "
            "Tu opinión es muy importante: ¿nos cuentas cómo fue tu último viaje?",
        "No leal con mala experiencia digital":
            f"Queremos que reservar y abordar sea fácil ({contexto}). "
            "¿Nos ayudas con 2 minutos de tu opinión sobre nuestros servicios en línea?",
        "Económica con poca comodidad":
            f"Estamos revisando la comodidad a bordo ({contexto}). "
            "¿Qué mejorarías de tu asiento, espacio, comida o entretenimiento?",
        "Otros en riesgo":
            f"Nos importa tu experiencia ({contexto}). "
            "Cuéntanos en una breve encuesta qué podemos hacer mejor.",
        SEG_MONITOREO:
            "Sin mensaje especial: este grupo se mantiene en monitoreo.",
    }
    return mensajes[segmento]


def valor_mas_frecuente(serie):
    return serie.mode().iloc[0] if not serie.empty else "—"


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """Muestra la pestaña "Decisiones": el plan final por segmento."""

    st.header("6. Decidir: plan de acción")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Qué haremos, con quién, cómo sabremos si funcionó y qué riesgos hay?**

    Conectamos cada segmento con su indicador, la evidencia, la acción, un
    responsable sugerido y un plan para medir el resultado.
    """)

    if df is None or "segmento" not in df.columns:
        st.warning(
            "Primero completa las pestañas **Predictiva** y **Prescriptiva** "
            "para generar los segmentos."
        )
        return


    resumen = resumen_segmentos(df)
    tasa_general = df["insatisfecho"].mean() * 100


    # ----------------------------------
    # TABLERO DE DECISIÓN
    # ----------------------------------

    st.subheader("Tablero de decisión")

    plan = pd.DataFrame({
        "Segmento": resumen["segmento"],
        "Pasajeros": resumen["pasajeros"],
        "% realmente insatisfechos": resumen["insatisfechos"].round(1),
        "Acción": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["accion"]),
        "Responsable sugerido": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["responsable"]),
        "Cómo medir el resultado": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["medicion"]),
        "Riesgo": resumen["segmento"].map(lambda s: PLAN_POR_SEGMENTO[s]["riesgo"]),
    })

    st.dataframe(plan, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Descargar plan de acción (CSV)",
        data=plan.to_csv(index=False).encode("utf-8-sig"),
        file_name="plan_de_accion.csv",
        mime="text/csv",
        key="descargar_plan_decisiones"
    )


    # ----------------------------------
    # DETALLE DE UN SEGMENTO
    # ----------------------------------

    st.subheader("Detalle de un segmento")

    segmentos_disponibles = [s for s in ORDEN_SEGMENTOS if s in set(resumen["segmento"])]

    segmento = st.selectbox(
        "Elige un segmento:",
        segmentos_disponibles,
        key="segmento_decisiones"
    )

    datos = df[df["segmento"] == segmento]
    fila = resumen[resumen["segmento"] == segmento].iloc[0]
    info = PLAN_POR_SEGMENTO[segmento]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pasajeros", f"{int(fila['pasajeros']):,}")
    col2.metric(
        "% realmente insatisfechos",
        f"{fila['insatisfechos']:.1f} %",
        delta=f"{fila['insatisfechos'] - tasa_general:+.1f} pts vs. total",
        delta_color="off"
    )
    col3.metric("Prob. media de insatisfacción", f"{fila['prob_media']:.1f} %")

    st.markdown(f"""
    **Evidencia (regla):** {info['regla']}

    **Acción recomendada:** {info['accion']}

    **Responsable sugerido:** {info['responsable']}

    **Cómo medir si funcionó:** {info['medicion']}

    **Riesgo o limitación:** {info['riesgo']}
    """)

    clase = valor_mas_frecuente(datos["Class"])
    viaje = valor_mas_frecuente(datos["Type of Travel"])

    st.markdown("#### Mensaje de ejemplo")
    st.info(mensaje_ejemplo(segmento, clase, viaje))

    st.caption(
        f"El mensaje usa lo más común en el segmento (clase {clase}, "
        f"viaje {viaje}). No promete descuentos, compensaciones ni beneficios: "
        "el dataset no dice cuáles existen. Defínelos con el negocio antes de enviarlo."
    )


    # ----------------------------------
    # CIERRE
    # ----------------------------------

    st.subheader("Antes de ejecutar")

    st.markdown("""
    - 🧪 **Prueba antes de escalar:** compara un grupo contactado con uno parecido sin contactar para saber si la acción realmente ayuda.
    - 🔗 **Los datos muestran asociaciones**, no causas.
    - 👀 **Revisión humana:** una persona debe validar segmentos, mensajes y presupuesto.
    - 🗂️ **Los datos son de práctica**: no hay identificador de pasajero ni fechas, así que este plan sirve para aprender el método.
    """)
