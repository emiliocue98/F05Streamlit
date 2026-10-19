# Proyecto: Analítica de marketing con Streamlit

Aplicación de Streamlit para analizar la satisfacción de pasajeros de una aerolínea (`Invistico_Airline.csv`) y convertir los datos en una decisión de marketing.

**Pregunta guía:** ¿Qué pasajeros están insatisfechos, qué lo explica y qué acción de recuperación o retención corresponde a cada tipo?

---

# Parte 1. Las 7 preguntas del proyecto

## 1. Entorno y estructura listos

**¿Qué hace un entorno virtual y por qué lo usas?**

Un entorno virtual es una carpeta (aquí `stream_train/`) con su propia copia de Python y de las librerías del proyecto. Lo uso porque:

- **Aísla las versiones:** este proyecto necesita `streamlit`, `pandas` y `scikit-learn` en versiones concretas, y otro proyecto de mi computadora podría necesitar otras sin que choquen.
- **Hace el proyecto reproducible:** `requirements.txt` guarda las versiones exactas (`pip freeze`), así que otra persona crea el mismo entorno con `pip install -r requirements.txt`.
- **No ensucia el Python del sistema.**

Se crea con `python -m venv stream_train`, se activa con `.\stream_train\Scripts\Activate.ps1` (Windows) y la app se ejecuta con `streamlit run app.py`. La carpeta del entorno no se sube a ningún repositorio.

Estructura del proyecto:

```text
Marketing_streamlit/
├── stream_train/          # entorno virtual (no se versiona)
├── data/
│   ├── Invistico_Airline.csv     # dataset actual
│   └── dataset_clientes_sinteticos_segmentos_marketing.csv   # dataset anterior, ya no se usa
├── tabs/
│   ├── entender.py        # 1. Entender
│   ├── limpiar.py         # 2. Limpiar
│   ├── descriptiva.py     # 3. Descriptiva
│   ├── predictiva.py      # 4. Predictiva
│   ├── prescriptiva.py    # 5. Prescriptiva
│   └── decisiones.py      # 6. Decisiones
├── app.py                 # configuración, carga de datos y pestañas
├── requirements.txt
└── project.md
```

`app.py` solo orquesta: carga el CSV una vez y pasa el dataframe a cada pestaña como argumento.

## 2. Primera ejecución con tu contexto

**¿Qué contexto elegiste, de dónde vienen los datos y qué pregunta quieres responder?**

- **Contexto:** una aerolínea que quiere mejorar la experiencia de sus pasajeros y evitar perder clientes insatisfechos.
- **Datos:** `data/Invistico_Airline.csv`, una encuesta de satisfacción con 129,880 pasajeros y 23 variables (perfil, tipo de viaje, clase, distancia, 14 calificaciones de servicio de 1 a 5, retrasos de salida y llegada, y el resultado `satisfaction`). Me llegó como archivo de práctica del curso. **No conozco con certeza su origen original ni cómo se hizo la encuesta**, y no tiene identificador de pasajero ni fechas. Hay que tratarlo como datos de práctica.
- **Pregunta:** ¿qué pasajeros están insatisfechos, qué lo explica y qué acción corresponde a cada tipo? Primera ejecución: la app abre en `http://localhost:8501` con las seis pestañas (Entender → Limpiar → Descriptiva → Predictiva → Prescriptiva → Decisiones).

Antes de este dataset el proyecto usaba un archivo sintético de clientes de una tienda. Lo cambié por el de aerolínea y adapté las seis pestañas.

## 3. Entender los datos

**¿Qué variables serán clave para tu decisión y cuáles te generan dudas?**

**Clave para la decisión**

- `satisfaction`: es el resultado que queremos explicar (54.7 % satisfechos, 45.3 % insatisfechos).
- `Customer Type` (leal / no leal): 61.6 % de satisfechos entre leales frente a 24.0 % entre no leales.
- `Class` y `Type of Travel`: Business 70.9 % de satisfechos; económica 39.4 %. Los viajeros de negocios en económica son el grupo más bajo (29.9 %).
- `Departure Delay in Minutes`: de 57.2 % de satisfechos sin retraso a 43.7 % con más de 60 minutos.
- Calificaciones de servicio, en especial entretenimiento a bordo, reserva en línea, soporte en línea, servicio a bordo y comodidad del asiento, que son las que más separan a satisfechos de insatisfechos.

**Me generan dudas**

- **Calificaciones iguales a 0:** la escala es de 1 a 5. Los 0 se concentran en pocas preguntas (asiento, horario, comida, entretenimiento), así que parecen «no aplica» o «sin respuesta», pero es una interpretación que debe confirmar el negocio.
- **`Arrival Delay in Minutes`:** 393 faltantes que no se explican por «sin retraso».
- **Las calificaciones y la satisfacción vienen de la misma encuesta:** sirven para explicar causas, pero no las tendríamos antes de que el pasajero responda.
- **No hay ID ni fecha:** no se puede contactar a nadie ni ver cambios en el tiempo.
- **Origen y método de la encuesta desconocidos:** no sé si la muestra representa a todos los pasajeros.

## 4. Un error o una decisión técnica

**Qué pasó, cómo lo diagnosticaste y qué aprendiste**

**Error: `KeyError: 'compras_12m'` al abrir la app tras cambiar de dataset.**

- **Qué pasó:** después de adaptar el código al dataset de aerolínea, el navegador seguía mostrando el error de una columna que solo existía en el dataset anterior.
- **Cómo lo diagnosticé:** busqué `compras_12m` en todos los archivos `.py` del proyecto y no apareció en ninguno. Si el código ya no la usaba, el error venía de código viejo que seguía cargado en el servidor en ejecución. Además, la app completa se ejecutó sin errores en una prueba aparte.
- **Solución:** reiniciar el servidor de Streamlit y refrescar el navegador (Ctrl+F5).
- **Qué aprendí:** primero comprobar que el error corresponde al código actual (buscar el nombre en los archivos) antes de editar nada. Un servidor ya abierto puede conservar módulos viejos en memoria. También aprendí que «el sitio no se puede abrir» puede significar simplemente que Streamlit no se está ejecutando, y que se verifica viendo si el puerto 8501 está activo.

**Decisión técnica: convertir el problema a «insatisfecho» (1) / «satisfecho» (0) y priorizar con validación cruzada.** Cada pasajero se puntúa con un modelo que no lo vio al entrenar, para que la evaluación sea honesta. El modelo es un gradient boosting; la importancia de variables se mide sobre una muestra de 10,000 filas para que la app responda en segundos.

## 5. Limpiar y describir

**Qué decidiste con los datos sucios y qué segmento o KPI llamó tu atención**

**Decisiones de limpieza** (siempre sobre una copia; el CSV original no se modifica):

| Problema | Decisión | Razón |
|---|---|---|
| 393 faltantes en retraso de llegada | **Conservar como faltante** | Solo 147 de 393 tienen retraso de salida = 0, así que no es un «sin retraso» implícito. Rellenar con 0 inventaría datos. Si hiciera falta, copiar el retraso de salida (correlación 0.97) documentándolo como suposición. |
| Calificaciones = 0 | **Convertir en faltante** al calcular KPI | «No aplica» no es lo mismo que «muy mala experiencia»; si se promedian como 0 distorsionan los promedios. |
| Duplicados | Sin acción | No hay duplicados exactos. |
| Valores atípicos (13.9 % en retraso de salida) | **Conservar y marcar**, no eliminar | Un retraso largo es un evento real y justo el que nos interesa. |
| Rangos inválidos | Sin cambios | Ninguna regla de validez se viola. |

**KPI y segmentos que llamaron mi atención**

- 54.7 % de satisfacción general.
- La diferencia entre **clientes leales (61.6 %) y no leales (24.0 %)**.
- El cruce **clase × motivo del viaje**: en Business los viajeros de negocios llegan a 72 % de satisfechos, en económica los de negocios caen a 30 %.
- El **retraso**: la satisfacción baja de 57.2 % sin retraso a 43.7 % con más de 60 minutos.
- La **brecha de calificación** más grande es entretenimiento a bordo (1.49 puntos), seguida de reserva en línea (1.13) y soporte en línea (1.02). Ubicación de la puerta y horario conveniente casi no separan a los dos grupos.

Estas diferencias son **descriptivas**: los grupos difieren en muchas cosas a la vez, así que no demuestran causas.

## 6. De los datos a la decisión

**Qué acción recomiendas, para quién y con qué evidencia**

Recomiendo empezar por dos grupos, porque tienen una acción clara y evidencia fuerte. Las reglas usan un umbral de riesgo del 70 % y un retraso largo de 60 minutos o más, ambos ajustables en la app.

| Para quién | Acción | Evidencia |
|---|---|---|
| **Pasajeros con retraso largo y alta probabilidad de insatisfacción** | Contacto proactivo de disculpa y seguimiento | Segmento de 4,736 pasajeros, **96.9 % realmente insatisfechos**. |
| **Clientes leales en riesgo** | Contacto personalizado para conocer su experiencia antes de perderlos | Son 81.7 % de la base; los leales insatisfechos son los más valiosos de recuperar. |

Después, de forma más exploratoria:

- **Clientes no leales con mala experiencia digital:** mejorar reserva, abordaje en línea y soporte.
- **Pasajeros de económica con poca comodidad:** investigar asiento, espacio, comida y entretenimiento.

Cómo se prioriza: el modelo ordena a los pasajeros por probabilidad de insatisfacción. Contactando al 20 % con mayor probabilidad se alcanza al **44 % de los insatisfechos, 2.2 veces mejor que elegir al azar**.

Cómo sabremos si funcionó: **prueba piloto con grupo de control** (contactados vs. un grupo parecido sin contactar) y comparar satisfacción y retención. Los mensajes no prometen descuentos ni compensaciones, porque el dataset no dice cuáles existen: eso lo define el negocio.

## 7. Reflexión de transferencia

**Qué cambiarías con datos reales: límites, sesgos y privacidad**

**Límites**

- Faltan un **identificador de pasajero** y la **fecha del vuelo**. Sin ellos no se puede contactar a nadie ni medir cambios.
- Los resultados son **asociaciones, no causas**. Una probabilidad no es una certeza: habrá pasajeros contactados que no lo necesitaban y otros omitidos que sí.
- Las calificaciones se miden junto con la satisfacción. En producción el modelo de priorización debería usar solo lo que se conoce *antes* de la encuesta (perfil, viaje, retrasos).
- No hay validación fuera de este archivo; habría que probarlo con datos nuevos de otro periodo.

**Sesgos**

- Quienes responden una encuesta no siempre representan a todos los pasajeros (sesgo de respuesta).
- Los 0 de las calificaciones podrían no significar lo mismo en todos los casos.
- El modelo podría tratar peor a ciertos grupos (edad, género, clase). Antes de usarlo habría que revisar su desempeño por grupo y dejar una persona que revise la lista.

**Privacidad**

- Con datos reales hay que cumplir la ley de protección de datos personales aplicable (por ejemplo, la mexicana) y tener consentimiento para contactar.
- Minimizar los datos: usar solo lo necesario, evitar variables sensibles como el género para decidir a quién contactar, y no mezclar identificadores con la encuesta sin necesidad.
- Proteger el archivo con la lista de pasajeros priorizados, porque identifica personas.

**Qué cambiaría concretamente:** agregar ID y fecha, validar el significado de los 0 con el negocio, entrenar solo con información previa al vuelo, auditar por grupo, definir beneficios reales con el negocio y medir la acción con un grupo de control antes de escalar.

---

# Parte 2. Guía técnica del proyecto

## Propósito y principios

Proyecto didáctico de una clase de dos horas: pasar de revisar datos a proponer una acción de marketing justificada. Se construye por etapas: **Entender → Limpiar → Describir → Predecir → Prescribir → Decidir**, cada una en su pestaña y su módulo.

- Inspeccionar e interpretar antes de modificar. Un faltante no es automáticamente un error; no se elimina ni imputa sin entender su significado y escribir la razón.
- Distinguir un resumen descriptivo de una conclusión de negocio. Asociación o puntaje de modelo no es causalidad.
- Los resultados del modelo apoyan la decisión; hay que explicar supuestos, errores, límites y revisión humana.
- Código legible para principiantes intermedios: etiquetas y comentarios en español, funciones con nombres explícitos.

## Dataset

Archivo: `data/Invistico_Airline.csv`.

- 129,880 registros y 23 columnas. Sin duplicados exactos.
- 393 faltantes en `Arrival Delay in Minutes`.
- 14 calificaciones de servicio en escala 1 a 5 que también contienen 0.
- Objetivo: `satisfaction` (`satisfied` 71,087 / `dissatisfied` 58,793). La pestaña Predictiva modela `insatisfecho` = 1 cuando es `dissatisfied`.

Columnas: `satisfaction`, `Gender`, `Customer Type`, `Age`, `Type of Travel`, `Class`, `Flight Distance`, las 14 calificaciones (`Seat comfort`, `Departure/Arrival time convenient`, `Food and drink`, `Gate location`, `Inflight wifi service`, `Inflight entertainment`, `Online support`, `Ease of Online booking`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness`, `Online boarding`), `Departure Delay in Minutes` y `Arrival Delay in Minutes`.

Los significados son interpretaciones a partir de los nombres de columna; no hay un diccionario de datos verificado.

## Arquitectura

`app.py` es solo orquestador: configura Streamlit, carga el CSV (con caché) y llama una función `mostrar(df)` por pestaña. Los datos fluyen así:

```text
df (crudo) → Limpiar → df_limpio → Descriptiva
                            └→ Predictiva → df_predicho → Prescriptiva → df_prescrito → Decisiones
```

Cada módulo recibe los datos como argumento; no recarga el dataset ni usa estado global oculto. Los widgets llevan `key` únicos.

## Qué hace cada pestaña

1. **Entender:** dimensiones, vista previa, diagnóstico (tipos, únicos, faltantes), duplicados y explorador de variables.
2. **Limpiar:** faltantes, duplicados, calificaciones = 0, rangos, atípicos y bitácora de decisiones con razón escrita. Devuelve `df_limpio`.
3. **Descriptiva:** KPI generales, satisfacción por segmento, cruce clase × motivo, brecha de calificación por servicio.
4. **Predictiva:** gradient boosting con validación cruzada; curva de ganancia, importancia de señales y lista de pasajeros por prioridad.
5. **Prescriptiva:** reglas visibles por segmento (la primera que se cumple) con umbrales ajustables.
6. **Decisiones:** tablero con segmento, evidencia, acción, responsable, medición y riesgo; mensaje de ejemplo y descarga del plan.

## Ejecución

```powershell
python -m venv stream_train
.\stream_train\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`. Si el sitio «no se puede abrir», probablemente Streamlit no se está ejecutando. Si aparece un error de una columna que ya no existe, reinicia el servidor.

## Guía para un agente de código

1. Leer este archivo primero y respetar el flujo por etapas.
2. Inspeccionar el árbol real del proyecto antes de cambiar código.
3. Hacer el cambio coherente más pequeño para la etapa actual.
4. Conservar el CSV original; la limpieza va en variables separadas y documentada.
5. Usar imports explícitos, nombres en español y `key` únicos en los widgets.
6. Verificar sintaxis e imports y ejecutar una comprobación local significativa.
7. No afirmar validez predictiva con este dataset de práctica. Señalar supuestos y riesgos de fuga de información.
