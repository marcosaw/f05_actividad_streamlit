import pandas as pd
import streamlit as st

RANGOS_VALIDACION = {
    "promedio_anterior": (0, 10),
    "porcentaje_asistencia": (0, 100),
    "materias_reprobadas": (0, None),
    "horas_estudio_semanal": (0, 168),
    "retencion": (0, 1),
}

OPCIONES_RANGO = {
    "Conservar": "No modificar los valores fuera de rango",
    "Convertir en faltante": "Tratarlos como faltantes (se pueden imputar después)",
    "Excluir filas": "Eliminar las filas que los contienen",
}


def _fuera_de_rango(df: pd.DataFrame) -> dict[str, pd.Series]:
    resultado = {}
    for columna, (minimo, maximo) in RANGOS_VALIDACION.items():
        if columna not in df:
            continue
        valores = pd.to_numeric(df[columna], errors="coerce")
        fuera = valores.lt(minimo)
        if maximo is not None:
            fuera |= valores.gt(maximo)
        resultado[columna] = fuera.fillna(False)
    return resultado


def _reporte_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Resume faltantes y valores fuera de rango por variable."""
    fuera = _fuera_de_rango(df)
    return pd.DataFrame(
        {
            "Variable": c,
            "Tipo": str(df[c].dtype),
            "Faltantes": int(df[c].isna().sum()),
            "Fuera de rango": int(fuera[c].sum()) if c in fuera else 0,
            "Valores únicos": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    )


def mostrar(df: pd.DataFrame) -> None:
    st.header("2. Limpiar los datos")
    st.markdown(
        "Inspecciona primero los problemas y decide cada tratamiento. "
        "El archivo original permanece intacto; los cambios se aplican a "
        "una copia de trabajo y se pueden rehacer."
    )

    if "df_limpio" not in st.session_state:
        st.session_state.df_limpio = df.copy()
        st.session_state.limpieza_acciones = ["Se creó una copia de trabajo sin cambios."]

    fuera = _fuera_de_rango(df)
    st.subheader("Diagnóstico antes del tratamiento")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas originales", f"{len(df):,}")
    c2.metric("Duplicados exactos", f"{int(df.duplicated().sum()):,}")
    c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Valores fuera de rango", f"{int(sum(m.sum() for m in fuera.values())):,}")
    st.dataframe(_reporte_calidad(df), use_container_width=True, hide_index=True)

    with st.expander("Rangos considerados válidos y registros afectados"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": c,
                    "Rango esperado": f"{RANGOS_VALIDACION[c][0]} a {RANGOS_VALIDACION[c][1] if RANGOS_VALIDACION[c][1] is not None else 'sin límite'}",
                    "Fuera de rango": int(m.sum()),
                }
                for c, m in fuera.items()
            ),
            use_container_width=True,
            hide_index=True,
        )
        filas_malas = df.loc[pd.concat(fuera.values(), axis=1).any(axis=1)] if fuera else df.iloc[0:0]
        if not filas_malas.empty:
            st.write("Registros con algún valor fuera de rango:")
            st.dataframe(filas_malas, use_container_width=True, hide_index=True)
        st.caption("Los extremos estadísticos válidos no se eliminan automáticamente.")

    st.subheader("Aplicar decisiones a la copia de trabajo")
    with st.form("formulario_limpieza"):
        quitar_duplicados = st.checkbox("Quitar filas duplicadas exactas", value=True, key="quitar_duplicados")
        tratamiento_rango = st.radio(
            "Valores fuera de rango",
            list(OPCIONES_RANGO),
            format_func=lambda o: f"{o}: {OPCIONES_RANGO[o]}",
            index=1,
            key="tratamiento_rango",
        )
        imputar = st.checkbox(
            "Imputar faltantes numéricos con la mediana de la variable",
            value=True,
            key="imputar_mediana",
            help="La mediana es robusta a valores extremos y conserva el número de alumnos.",
        )
        aplicar = st.form_submit_button("Aplicar limpieza seleccionada")

    if aplicar:
        limpio = df.copy()
        acciones = []
        if quitar_duplicados:
            antes = len(limpio)
            limpio = limpio.drop_duplicates().reset_index(drop=True)
            acciones.append(f"Se quitaron {antes - len(limpio)} filas duplicadas exactas.")
        if tratamiento_rango != "Conservar":
            malos = _fuera_de_rango(limpio)
            if tratamiento_rango == "Excluir filas":
                mascara = pd.concat(malos.values(), axis=1).any(axis=1) if malos else pd.Series(False, index=limpio.index)
                cantidad = int(mascara.sum())
                limpio = limpio.loc[~mascara].reset_index(drop=True)
                acciones.append(f"Se excluyeron {cantidad} filas con valores fuera de rango.")
            else:
                total = 0
                for columna, mascara in malos.items():
                    limpio[columna] = pd.to_numeric(limpio[columna], errors="coerce").mask(mascara)
                    total += int(mascara.sum())
                acciones.append(f"Se convirtieron {total} valores fuera de rango en faltantes.")
        if imputar:
            for columna in limpio.select_dtypes(include="number").columns:
                if columna == "retencion":
                    continue
                cantidad = int(limpio[columna].isna().sum())
                if cantidad:
                    mediana = limpio[columna].median()
                    limpio[columna] = limpio[columna].fillna(mediana)
                    acciones.append(f"Se imputaron {cantidad} faltantes de {columna} con la mediana ({mediana:.2f}).")
        if not acciones:
            acciones.append("No se hicieron cambios; se conservaron faltantes y duplicados.")
        st.session_state.df_limpio = limpio
        st.session_state.limpieza_acciones = acciones
        st.success("Se actualizó la copia de trabajo.")

    df_limpio = st.session_state.df_limpio
    st.subheader("Resultado y bitácora")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas de trabajo", f"{len(df_limpio):,}")
    c2.metric("Faltantes restantes", f"{int(df_limpio.isna().sum().sum()):,}")
    c3.metric("Duplicados restantes", f"{int(df_limpio.duplicated().sum()):,}")
    c4.metric("Fuera de rango restantes", f"{int(sum(m.sum() for m in _fuera_de_rango(df_limpio).values())):,}")
    for accion in st.session_state.limpieza_acciones:
        st.write(f"- {accion}")

    st.download_button(
        "Descargar copia de trabajo (CSV)",
        data=df_limpio.to_csv(index=False).encode("utf-8-sig"),
        file_name="alumnos_limpios.csv",
        mime="text/csv",
        key="descargar_limpieza",
    )
