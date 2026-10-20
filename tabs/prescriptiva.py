import pandas as pd
import streamlit as st

from tabs.modelo import entrenar_y_evaluar, validar

NIVELES = ["Riesgo alto", "Riesgo medio", "Riesgo bajo"]

# Intensidad de la intervención según el nivel de riesgo
PLAN_POR_NIVEL = {
    "Riesgo alto": {
        "canal": "Llamada o WhatsApp personal del tutor",
        "responsable": "Coordinación académica / tutoría",
        "base": "Contacto personal prioritario y plan de acompañamiento individual",
    },
    "Riesgo medio": {
        "canal": "Correo institucional + cita con tutor",
        "responsable": "Tutor de grupo",
        "base": "Seguimiento automatizado y cita breve con tutor",
    },
    "Riesgo bajo": {
        "canal": "Boletín institucional",
        "responsable": "Servicios escolares",
        "base": "Acompañamiento estándar y comunicación de recursos disponibles",
    },
}

# Apoyo específico según el factor de riesgo más claro del alumno
APOYO_POR_MOTIVO = {
    "Pago pendiente": "orientación financiera: plan de pagos o revisión de beca",
    "Materias reprobadas": "tutoría académica en las materias reprobadas",
    "Baja asistencia": "seguimiento de asistencia y revisión de barreras (horarios, transporte, salud)",
    "Bajo promedio": "asesoría en métodos de estudio y nivelación",
    "Baja participación": "invitación a actividades y mentoría entre pares",
    "Sin factor dominante": "acompañamiento general",
}

MENSAJES = {
    "Riesgo alto": "Hola. Queremos saber cómo vas y apoyarte en lo que necesites. ¿Platicamos esta semana? Tenemos opciones de apoyo: {apoyo}.",
    "Riesgo medio": "Hola. Tu tutor está disponible para ayudarte a que el semestre vaya bien. Puedes agendar una cita; ofrecemos {apoyo}.",
    "Riesgo bajo": "Hola. Gracias por continuar con nosotros. Recuerda que cuentas con tutorías y servicios de apoyo cuando los necesites.",
}


def _num(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _motivo_principal(df: pd.DataFrame) -> pd.Series:
    """Primer factor de riesgo observable; el orden define la prioridad."""
    motivo = pd.Series("Sin factor dominante", index=df.index, dtype="object")
    reglas = [
        ("Baja participación", df.get("participacion_actividades", pd.Series(index=df.index, dtype=object)).eq("Baja")),
        ("Bajo promedio", _num(df, "promedio_anterior").lt(7)),
        ("Baja asistencia", _num(df, "porcentaje_asistencia").lt(75)),
        ("Materias reprobadas", _num(df, "materias_reprobadas").ge(2)),
        ("Pago pendiente", df.get("pago_al_dia", pd.Series(index=df.index, dtype=object)).eq("No")),
    ]
    # Se aplican de menor a mayor prioridad para que la última regla prevalezca
    for nombre, condicion in reglas:
        motivo.loc[condicion.fillna(False)] = nombre
    return motivo


def construir_recomendaciones(
    df: pd.DataFrame, riesgo: pd.Series, umbral_alto: float = 0.70, umbral_medio: float = 0.40
) -> pd.DataFrame:
    """Asigna nivel de riesgo, acción, canal y mensaje a cada alumno."""
    resultado = df.copy()
    resultado["probabilidad_abandono"] = riesgo.reindex(resultado.index)
    prob = resultado["probabilidad_abandono"]

    nivel = pd.Series("Riesgo bajo", index=resultado.index, dtype="object")
    nivel.loc[prob.ge(umbral_medio)] = "Riesgo medio"
    nivel.loc[prob.ge(umbral_alto)] = "Riesgo alto"
    motivo = _motivo_principal(resultado)
    apoyo = motivo.map(APOYO_POR_MOTIVO)

    resultado["segmento_recomendado"] = nivel
    resultado["factor_principal"] = motivo
    resultado["evidencia_regla"] = (
        "Probabilidad estimada de abandono " + (prob * 100).round(0).astype("Int64").astype(str)
        + "%; factor principal: " + motivo.str.lower() + "."
    )
    resultado["accion_recomendada"] = nivel.map(lambda n: PLAN_POR_NIVEL[n]["base"]) + ": " + apoyo
    resultado["canal_a_validar"] = nivel.map(lambda n: PLAN_POR_NIVEL[n]["canal"])
    resultado["responsable_sugerido"] = nivel.map(lambda n: PLAN_POR_NIVEL[n]["responsable"])
    resultado["mensaje_borrador"] = [MENSAJES[n].format(apoyo=a) for n, a in zip(nivel, apoyo)]
    resultado["siguiente_paso"] = "Validar la lista con coordinación y aplicar la intervención con un grupo de control."
    resultado["riesgo_medicion"] = "Probabilidad estimada con datos sintéticos; medir el efecto real con grupo de control."
    return resultado


def mostrar(df: pd.DataFrame) -> None:
    st.header("5. Estadística prescriptiva")
    st.markdown(
        "Traduce la probabilidad de abandono de la pestaña anterior en una **acción "
        "de acompañamiento**. El nivel de riesgo define la intensidad y el canal; el "
        "factor principal de cada alumno (asistencia, reprobación, pagos, etc.) define "
        "el tipo de apoyo. Son reglas explicables y no sustituyen el criterio del tutor."
    )
    if df.empty:
        st.warning("No hay alumnos para segmentar.")
        return
    error = validar(df)
    if error:
        st.error(error)
        return
    try:
        riesgo = entrenar_y_evaluar(df)["riesgo"]
    except (ModuleNotFoundError, ValueError) as error:
        st.error(f"No se pudo calcular el riesgo de abandono: {error}")
        return

    a, b = st.columns(2)
    alto = a.slider("Umbral de riesgo alto (probabilidad ≥)", 0.50, 0.95, 0.70, 0.05, key="umbral_alto")
    medio = b.slider("Umbral de riesgo medio (probabilidad ≥)", 0.10, 0.60, 0.40, 0.05, key="umbral_medio")
    if medio >= alto:
        st.warning("El umbral medio debe ser menor que el alto.")
        return

    recomendaciones = construir_recomendaciones(df, riesgo, alto, medio)
    st.session_state.recomendaciones = recomendaciones

    conteos = (
        recomendaciones["segmento_recomendado"].value_counts()
        .reindex(NIVELES, fill_value=0)
        .rename_axis("Nivel de riesgo").rename("Alumnos").reset_index()
    )
    st.subheader("Tamaño de cada grupo")
    st.bar_chart(conteos.set_index("Nivel de riesgo"))

    st.subheader("Acciones prescritas por nivel de riesgo")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Nivel": nivel,
                    "Regla": regla,
                    "Acción base": PLAN_POR_NIVEL[nivel]["base"],
                    "Canal recomendado": PLAN_POR_NIVEL[nivel]["canal"],
                    "Responsable": PLAN_POR_NIVEL[nivel]["responsable"],
                }
                for nivel, regla in zip(
                    NIVELES,
                    [
                        f"Probabilidad ≥ {alto:.0%}",
                        f"{medio:.0%} ≤ probabilidad < {alto:.0%}",
                        f"Probabilidad < {medio:.0%}",
                    ],
                )
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Apoyo específico según el factor principal")
    st.caption("Si varios factores coinciden, prevalece el primero de esta lista (de mayor a menor prioridad).")
    st.dataframe(
        pd.DataFrame(
            [
                {"Prioridad": i, "Factor": factor, "Condición": condicion, "Apoyo": APOYO_POR_MOTIVO[factor]}
                for i, (factor, condicion) in enumerate(
                    [
                        ("Pago pendiente", "pago_al_dia = No"),
                        ("Materias reprobadas", "materias_reprobadas ≥ 2"),
                        ("Baja asistencia", "porcentaje_asistencia < 75"),
                        ("Bajo promedio", "promedio_anterior < 7"),
                        ("Baja participación", "participacion_actividades = Baja"),
                        ("Sin factor dominante", "Ninguna de las anteriores"),
                    ],
                    start=1,
                )
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Muestra de recomendaciones")
    nivel = st.selectbox("Ver nivel", NIVELES, key="nivel_muestra_prescriptiva")
    muestra = recomendaciones.loc[
        recomendaciones["segmento_recomendado"].eq(nivel),
        ["id_alumno", "probabilidad_abandono", "factor_principal", "accion_recomendada", "mensaje_borrador"],
    ].sort_values("probabilidad_abandono", ascending=False).head(25)
    st.dataframe(
        muestra.style.format({"probabilidad_abandono": "{:.0%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Los mensajes ofrecen apoyo y no prometen becas ni beneficios que el dataset no documenta.")
