import pandas as pd
import streamlit as st

from tabs.modelo import entrenar_y_evaluar, validar


def mostrar(df: pd.DataFrame) -> None:
    st.header("4. Estadística predictiva")
    st.markdown(
        "Estimamos la **probabilidad de abandono** de cada alumno con regresión "
        "logística, un modelo interpretable que permite ver qué variables pesan más. "
        "Se compara con un Random Forest para comprobar que no se pierde capacidad "
        "predictiva. Es un ejercicio sobre datos históricos: ordena y prioriza, "
        "no confirma quién abandonará el próximo periodo."
    )

    if df.empty:
        st.warning("No hay registros para analizar.")
        return
    error = validar(df)
    if error:
        st.error(error)
        return

    try:
        resultado = entrenar_y_evaluar(df)
    except ModuleNotFoundError:
        st.error("Falta scikit-learn. Instala las dependencias del proyecto.")
        return
    except ValueError as error:
        st.error(f"No se pudo entrenar el modelo con estos datos: {error}")
        return

    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    st.subheader("Comparación de modelos")
    comparacion = resultado["comparacion"]
    st.dataframe(
        comparacion.style.format(
            {
                "AUC validación cruzada (media)": "{:.3f}",
                "AUC validación cruzada (desv.)": "{:.3f}",
                "AUC prueba (25%)": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "AUC: 0.5 equivale a ordenar al azar y 1.0 es una separación perfecta. "
        "La validación cruzada (5 pliegues) es más estable que una sola partición "
        "con tan pocos alumnos. Si ambos modelos rinden parecido, se prefiere la "
        "regresión logística por ser explicable."
    )

    st.subheader("Desempeño de la regresión logística en el conjunto de prueba")
    umbral = st.slider(
        "Umbral de probabilidad para marcar a un alumno como 'en riesgo'",
        0.20,
        0.80,
        0.50,
        0.05,
        key="umbral_predictiva",
        help="Un umbral bajo detecta más abandonos pero genera más falsas alarmas.",
    )
    y_real = resultado["y_prueba"]
    y_pred = (resultado["prob_prueba"]["Regresión logística"] >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    exactitud = (tp + tn) / len(y_real)
    base = max(y_real.mean(), 1 - y_real.mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exactitud", f"{exactitud:.1%}", help=f"Referencia: predecir siempre la clase mayoritaria acierta {base:.1%}.")
    m2.metric(
        "Abandonos detectados (recall)",
        f"{recall_score(y_real, y_pred, zero_division=0):.1%}",
        help="De los alumnos que realmente abandonaron, cuántos quedaron marcados en riesgo.",
    )
    m3.metric(
        "Precisión de la alerta",
        f"{precision_score(y_real, y_pred, zero_division=0):.1%}",
        help="De los alumnos marcados en riesgo, cuántos realmente abandonaron.",
    )
    m4.metric("Alumnos en la prueba", f"{len(y_real):,}")

    st.dataframe(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Real: permanece", "Real: abandona"],
            columns=["Predicho: permanece", "Predicho: abandona"],
        ),
        use_container_width=True,
    )
    st.caption(
        f"Falsas alarmas: {fp}. Abandonos no detectados: {fn}. En retención suele "
        "ser preferible tolerar falsas alarmas (un apoyo innecesario es barato) que "
        "dejar pasar abandonos."
    )

    st.subheader("Importancia de las variables (coeficientes)")
    coeficientes = resultado["coeficientes"]
    st.bar_chart(coeficientes.set_index("Variable")["Coeficiente"])
    st.dataframe(
        coeficientes.style.format({"Coeficiente": "{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Las variables numéricas están estandarizadas, por lo que sus coeficientes son "
        "comparables. Positivo: aumenta la probabilidad de abandono. Negativo: la reduce. "
        "Son asociaciones, no causas."
    )

    with st.expander("Cómo se hizo esta evaluación"):
        st.markdown(
            f"- **Variable a predecir:** abandono = 1 − `retencion` "
            f"({resultado['tasa_abandono']:.1%} de los alumnos).\n"
            f"- **Variables usadas:** {', '.join(resultado['variables'])}. "
            "Se excluyen el identificador y el resultado.\n"
            "- **Preparación:** nulos numéricos con la mediana (nulos categóricos como "
            "'(sin dato)'), estandarización para la regresión logística y codificación "
            "one-hot de las variables categóricas.\n"
            "- **Desbalance:** se usa `class_weight` balanceado para que el modelo no "
            "ignore la clase minoritaria (abandono).\n"
            f"- **Validación:** partición aleatoria estratificada ({resultado['n_entrenamiento']} "
            f"para entrenar, {resultado['n_prueba']} para probar) y validación cruzada de 5 pliegues.\n"
            "- **Riesgo por alumno:** la pestaña 5 usa probabilidades calculadas fuera de "
            "muestra (cada alumno es evaluado por un modelo que no lo vio).\n"
            "- **Límites:** 250 registros sintéticos, evaluación aleatoria (no temporal) y "
            "sin variables como situación familiar o salud. Validar con datos reales antes de usarse."
        )
