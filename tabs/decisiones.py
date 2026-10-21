import pandas as pd
import streamlit as st

from tabs.prescriptiva import APOYO_POR_MOTIVO, PLAN_POR_NIVEL

CONTROL = "Control (sin intervención)"
TODOS_LOS_MOTIVOS = "Todos los motivos"


def _alumnos(n: int) -> str:
    return f"{n} alumno" if n == 1 else f"{n} alumnos"


def _formato_tasa(valores: pd.Series) -> str:
    tasa = pd.to_numeric(valores, errors="coerce").mean()
    return f"{tasa:.1%}" if pd.notna(tasa) else "N/D"


def _crear_plan(segmento: pd.DataFrame, porcentaje_piloto: int, porcentaje_control: int) -> pd.DataFrame:
    """Selecciona un piloto reproducible y separa control de tratamiento."""
    if segmento.empty:
        return segmento.copy()
    n_piloto = max(1, round(len(segmento) * porcentaje_piloto / 100))
    piloto = segmento.sample(n=n_piloto, random_state=42).copy()
    n_control = (
        min(len(piloto) - 1, max(1, round(len(piloto) * porcentaje_control / 100)))
        if len(piloto) >= 2
        else 0
    )
    grupos = pd.Series("Tratamiento", index=piloto.index)
    if n_control:
        indices_control = piloto.sample(n=n_control, random_state=43).index
        grupos.loc[indices_control] = CONTROL
    piloto["grupo_experimento"] = grupos
    piloto["kpi_primario"] = "Tasa de retención al cierre del periodo vs. grupo de control"
    piloto["ventana_medicion"] = "Definir antes del lanzamiento; aplicar igual a ambos grupos"
    return piloto


def mostrar(df: pd.DataFrame) -> None:
    st.header("6. Toma de decisiones")
    st.markdown(
        "Conecta un KPI, un nivel de riesgo, la evidencia, una acción de apoyo y un "
        "plan de medición. Esta pantalla prepara una propuesta revisable; no envía mensajes."
    )

    recomendaciones = st.session_state.get("recomendaciones")
    if recomendaciones is None or recomendaciones.empty:
        st.info("No hay recomendaciones aún. Revisa primero la pestaña Prescriptiva.")
        return

    objetivo = "retencion"
    filas = []
    for nombre, grupo in recomendaciones.groupby("segmento_recomendado", dropna=False):
        fila = {
            "Nivel de riesgo": nombre,
            "Alumnos": len(grupo),
            "Participación": len(grupo) / len(recomendaciones),
            "Acción": grupo["accion_recomendada"].mode().iat[0],
            "Evidencia ejemplo": grupo["evidencia_regla"].iat[0],
            "Responsable": grupo["responsable_sugerido"].iat[0],
            "Riesgo / medición": grupo["riesgo_medicion"].iat[0],
        }
        if objetivo in grupo:
            fila["Retención histórica"] = _formato_tasa(grupo[objetivo])
        filas.append(fila)
    resumen = pd.DataFrame(filas).sort_values("Alumnos", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Alumnos en análisis", f"{len(recomendaciones):,}")
    c2.metric("Niveles de riesgo", f"{recomendaciones['segmento_recomendado'].nunique():,}")
    c3.metric("Retención observada", _formato_tasa(recomendaciones[objetivo]) if objetivo in recomendaciones else "N/D")

    st.subheader("Resumen para revisión")
    st.dataframe(
        resumen.style.format({"Participación": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "La retención histórica por nivel es descriptiva: no estima el efecto de "
        "la intervención. Para medir el efecto, compara tratamiento y control."
    )

    st.subheader("Seleccionar un nivel de riesgo")
    niveles = resumen["Nivel de riesgo"].tolist()
    nivel_elegido = st.selectbox("Nivel de riesgo prioritario", niveles, key="segmento_decisiones")
    grupo_nivel = recomendaciones.loc[recomendaciones["segmento_recomendado"].eq(nivel_elegido)].copy()

    # Dos alumnos con el mismo nivel de riesgo pueden necesitar apoyos distintos:
    # el motivo principal decide el tipo de apoyo y el mensaje.
    conteo_motivos = grupo_nivel["factor_principal"].value_counts()
    opciones_motivo = [TODOS_LOS_MOTIVOS] + conteo_motivos.index.tolist()
    motivo_elegido = st.selectbox(
        "Motivo principal",
        opciones_motivo,
        format_func=lambda m: f"{m} ({_alumnos(len(grupo_nivel) if m == TODOS_LOS_MOTIVOS else conteo_motivos[m])})",
        key=f"motivo_decisiones_{nivel_elegido}",
        help="Filtra el nivel de riesgo por la causa más clara de cada alumno: el apoyo y el mensaje cambian según el motivo.",
    )
    if motivo_elegido == TODOS_LOS_MOTIVOS:
        st.caption(
            "Elige un motivo para ver el apoyo y el mensaje específicos. Con todos los motivos "
            "juntos, el apoyo varía por alumno."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Motivo principal": conteo_motivos.index,
                    "Alumnos": conteo_motivos.values,
                    "Apoyo sugerido": [APOYO_POR_MOTIVO.get(m, "") for m in conteo_motivos.index],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        grupo = grupo_nivel
    else:
        grupo = grupo_nivel.loc[grupo_nivel["factor_principal"].eq(motivo_elegido)].copy()

    mensaje_base = grupo.iloc[0]
    # Con un solo motivo, todos los alumnos comparten apoyo y mensaje; con todos los motivos
    # solo la intensidad (canal y responsable) es común al nivel de riesgo.
    if motivo_elegido == TODOS_LOS_MOTIVOS:
        accion = PLAN_POR_NIVEL[nivel_elegido]["base"]
    else:
        accion = mensaje_base["accion_recomendada"]
    if nivel_elegido == "Riesgo alto":
        st.error(f"Acción inmediata: {accion}")
    elif nivel_elegido == "Riesgo medio":
        st.warning(f"Acción de seguimiento: {accion}")
    else:
        st.success(f"Acción estándar: {accion}")
    if motivo_elegido != TODOS_LOS_MOTIVOS:
        st.write("**Mensaje sugerido (borrador):**")
        st.info(mensaje_base["mensaje_borrador"])
    st.write(f"**Canal recomendado:** {mensaje_base['canal_a_validar']}")
    st.write(f"**Responsable sugerido:** {mensaje_base['responsable_sugerido']}")

    st.subheader("Diseñar un piloto")
    c4, c5 = st.columns(2)
    porcentaje_piloto = c4.slider("Porcentaje del grupo para el piloto", 10, 100, 100, 5, key="porcentaje_piloto")
    porcentaje_control = c5.slider("Porcentaje del piloto reservado como control", 5, 40, 20, 5, key="porcentaje_control")
    confirmar_permiso = st.checkbox(
        "Confirmo que la lista fue revisada por coordinación académica y que el contacto "
        "cumple el aviso de privacidad de la institución.",
        key="confirmar_permiso_decisiones",
    )

    plan = _crear_plan(grupo, porcentaje_piloto, porcentaje_control)
    tratamiento = plan[plan["grupo_experimento"].eq("Tratamiento")]
    control = plan[plan["grupo_experimento"].eq(CONTROL)]
    c6, c7, c8 = st.columns(3)
    c6.metric("Tamaño del grupo elegido", f"{len(grupo):,}")
    c7.metric("Tratamiento", f"{len(tratamiento):,}")
    c8.metric("Control", f"{len(control):,}")
    if len(plan) < 2 or control.empty or tratamiento.empty:
        st.warning(
            "Este grupo es demasiado pequeño para formar tratamiento y control. "
            "Amplía el grupo o no interpretes el piloto como medición de efecto."
        )

    st.subheader("Medición propuesta")
    st.markdown(
        "1. Definir la ventana de medición y el KPI primario (retención al cierre del periodo) antes de iniciar.\n"
        "2. Aplicar el apoyo solo al grupo tratamiento; mantener el control sin esa intervención.\n"
        "3. Comparar la tasa de retención entre grupos con la misma ventana y documentar el tamaño de muestra.\n"
        "4. Cuidar el tono de los mensajes y la frecuencia de contacto; evitar estigmatizar a los alumnos marcados.\n"
        "5. Revisar cada caso con su tutor antes de asignar el apoyo: el motivo principal es una regla, no un diagnóstico."
    )
    st.caption(
        "Esta aplicación no envía comunicaciones. El CSV es un plan de trabajo; "
        "la confirmación marcada aquí representa una revisión del usuario, no "
        "un dato verificado por la aplicación."
    )

    columnas_exportacion = [
        c for c in [
            "id_alumno", "segmento_recomendado", "factor_principal", "probabilidad_abandono", "evidencia_regla",
            "accion_recomendada", "canal_a_validar", "mensaje_borrador", "responsable_sugerido",
            "siguiente_paso", "riesgo_medicion", "grupo_experimento", "kpi_primario",
            "ventana_medicion",
        ] if c in plan.columns
    ]
    st.download_button(
        "Descargar plan de piloto (CSV)",
        data=plan[columnas_exportacion].to_csv(index=False).encode("utf-8-sig"),
        file_name="plan_piloto_retencion.csv",
        mime="text/csv",
        disabled=(not confirmar_permiso) or len(plan) < 2 or control.empty or tratamiento.empty,
        help="Confirma la revisión y asegúrate de que el piloto tenga tratamiento y control para habilitar la descarga.",
        key="descargar_plan_decisiones",
    )
