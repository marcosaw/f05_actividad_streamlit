import pandas as pd
import streamlit as st

CATEGORICAS = ["modalidad_estudio", "nivel_socioeconomico", "pago_al_dia", "participacion_actividades"]
NUMERICAS = ["porcentaje_asistencia", "promedio_anterior", "materias_reprobadas", "horas_estudio_semanal"]


def _numero(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _media(serie: pd.Series, formato: str) -> str:
    valor = serie.mean()
    return formato.format(valor) if pd.notna(valor) else "N/D"


def _tabla_retencion(grupos: pd.Series, retencion: pd.Series, etiqueta: str) -> pd.DataFrame:
    tabla = (
        pd.DataFrame({etiqueta: grupos.astype("object").fillna("(sin dato)"), "retencion": retencion})
        .dropna(subset=["retencion"])
        .groupby(etiqueta, observed=True)
        .agg(alumnos=("retencion", "size"), tasa_retencion=("retencion", "mean"))
        .reset_index()
    )
    tabla[etiqueta] = tabla[etiqueta].astype(str)
    tabla["tasa_abandono"] = 1 - tabla["tasa_retencion"]
    return tabla


def mostrar(df: pd.DataFrame) -> None:
    st.header("3. Estadística descriptiva")
    st.markdown(
        "Resume lo observado en el conjunto actual. Las tasas muestran su "
        "denominador y describen asociaciones; no explican causalidad."
    )

    if df.empty:
        st.warning("No hay registros para resumir.")
        return

    retencion = _numero(df, "retencion")
    retencion = retencion.where(retencion.isin([0, 1]))
    validos = int(retencion.notna().sum())
    tasa = retencion.mean()

    st.subheader("Indicadores principales")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alumnos", f"{len(df):,}")
    m2.metric("Tasa de retención", f"{tasa:.1%}" if pd.notna(tasa) else "N/D", help=f"Sobre {validos:,} alumnos con resultado válido.")
    m3.metric("Tasa de abandono", f"{1 - tasa:.1%}" if pd.notna(tasa) else "N/D")
    m4.metric("Alumnos que abandonan", f"{int((retencion == 0).sum()):,}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Promedio anterior", _media(_numero(df, "promedio_anterior"), "{:.2f} / 10"))
    m6.metric("Asistencia media", _media(_numero(df, "porcentaje_asistencia"), "{:.1f}%"))
    m7.metric("Materias reprobadas (media)", _media(_numero(df, "materias_reprobadas"), "{:.2f}"))
    m8.metric("Horas de estudio semanales", _media(_numero(df, "horas_estudio_semanal"), "{:.1f}"))

    st.subheader("Retención por segmento")
    opciones = [c for c in CATEGORICAS if c in df]
    if opciones:
        variable = st.selectbox("Segmentar por:", opciones, key="segmento_descriptiva")
        tabla = _tabla_retencion(df[variable], retencion, variable)
        st.bar_chart(tabla.set_index(variable)["tasa_retencion"].mul(100).rename("Retención (%)"))
        st.dataframe(
            tabla.style.format({"tasa_retencion": "{:.1%}", "tasa_abandono": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        mejor, peor = tabla.loc[tabla["tasa_retencion"].idxmax()], tabla.loc[tabla["tasa_retencion"].idxmin()]
        st.caption(
            f"Mayor retención: {mejor[variable]} ({mejor['tasa_retencion']:.1%}). "
            f"Menor retención: {peor[variable]} ({peor['tasa_retencion']:.1%}). "
            "Compara también el número de alumnos de cada grupo antes de concluir."
        )

    st.subheader("Retención según indicadores académicos")
    numericas = [c for c in NUMERICAS if c in df]
    if numericas:
        columna = st.selectbox("Variable numérica:", numericas, key="numerica_descriptiva")
        serie = _numero(df, columna)
        if serie.nunique() > 1:
            try:
                rangos = pd.qcut(serie, q=4, duplicates="drop")
            except ValueError:
                rangos = pd.cut(serie, bins=3)
            tabla_num = _tabla_retencion(rangos, retencion, "rango")
            st.bar_chart(tabla_num.set_index("rango")["tasa_retencion"].mul(100).rename("Retención (%)"))
            st.dataframe(
                tabla_num.style.format({"tasa_retencion": "{:.1%}", "tasa_abandono": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Esta variable no tiene variación suficiente para agruparse.")
        st.caption("Los rangos son cuartiles de la variable elegida. Los valores faltantes se omiten.")
