"""Modelo de riesgo de abandono compartido por las pestañas predictiva y prescriptiva."""
import numpy as np
import pandas as pd
import streamlit as st

OBJETIVO = "retencion"  # 1 = permanece, 0 = abandona
COLUMNAS_EXCLUIDAS = {"id_alumno", OBJETIVO}
MIN_POR_CLASE = 10


def validar(df: pd.DataFrame) -> str | None:
    """Devuelve un mensaje de error si el dataset no permite entrenar, o None."""
    if OBJETIVO not in df:
        return f"Falta la columna `{OBJETIVO}` (1 = permanece, 0 = abandona)."
    objetivo = pd.to_numeric(df[OBJETIVO], errors="coerce")
    conteo = objetivo[objetivo.isin([0, 1])].value_counts()
    if set(conteo.index) != {0, 1}:
        return "La variable `retencion` debe contener alumnos con valor 0 y con valor 1."
    if int(conteo.min()) < MIN_POR_CLASE:
        return (
            f"Se necesitan al menos {MIN_POR_CLASE} alumnos de cada clase para "
            "validar el modelo con validación cruzada."
        )
    return None


def _preprocesador(X: pd.DataFrame, escalar: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X.columns if c not in numericas]
    pasos_num = [("completar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    transformadores = [("num", Pipeline(pasos_num), numericas)]
    if categoricas:
        transformadores.append(
            (
                "cat",
                Pipeline(
                    [
                        ("completar", SimpleImputer(strategy="constant", fill_value="(sin dato)")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            )
        )
    return ColumnTransformer(transformadores, remainder="drop")


def _modelos(X: pd.DataFrame) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return {
        "Regresión logística": Pipeline(
            [
                ("preparar", _preprocesador(X, escalar=True)),
                ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preparar", _preprocesador(X, escalar=False)),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


@st.cache_data(show_spinner="Entrenando y validando los modelos...")
def entrenar_y_evaluar(df: pd.DataFrame) -> dict:
    """Entrena regresión logística (modelo principal) y Random Forest (comparación).

    La variable que se predice es el abandono (1 - retencion). Devuelve métricas
    de validación cruzada y de un conjunto de prueba, los coeficientes del modelo
    lineal y el riesgo de abandono de cada alumno estimado fuera de muestra.
    """
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_predict,
        cross_val_score,
        train_test_split,
    )

    objetivo = pd.to_numeric(df[OBJETIVO], errors="coerce")
    etiquetado = objetivo.isin([0, 1])
    columnas = [c for c in df.columns if c not in COLUMNAS_EXCLUIDAS]
    X = df.loc[etiquetado, columnas]
    y = (1 - objetivo[etiquetado]).astype(int)  # 1 = abandona

    modelos = _modelos(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    comparacion, prueba = [], {}
    for nombre, pipeline in modelos.items():
        auc_cv = cross_val_score(clone(pipeline), X, y, cv=cv, scoring="roc_auc")
        ajustado = clone(pipeline).fit(X_ent, y_ent)
        prob = ajustado.predict_proba(X_prueba)[:, 1]
        prueba[nombre] = prob
        comparacion.append(
            {
                "Modelo": nombre,
                "AUC validación cruzada (media)": float(auc_cv.mean()),
                "AUC validación cruzada (desv.)": float(auc_cv.std()),
                "AUC prueba (25%)": float(roc_auc_score(y_prueba, prob)),
            }
        )

    # Coeficientes del modelo lineal ajustado con todos los datos etiquetados
    principal = clone(modelos["Regresión logística"]).fit(X, y)
    nombres = principal.named_steps["preparar"].get_feature_names_out()
    coeficientes = (
        pd.DataFrame(
            {
                "Variable": [n.split("__", 1)[1] for n in nombres],
                "Coeficiente": principal.named_steps["modelo"].coef_[0],
            }
        )
        .assign(Magnitud=lambda t: t["Coeficiente"].abs())
        .sort_values("Magnitud", ascending=False)
        .drop(columns="Magnitud")
        .reset_index(drop=True)
    )

    # Riesgo por alumno: fuera de muestra para los etiquetados, modelo final para el resto
    riesgo = pd.Series(np.nan, index=df.index, dtype="float64")
    riesgo.loc[etiquetado] = cross_val_predict(
        clone(modelos["Regresión logística"]), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    if (~etiquetado).any():
        riesgo.loc[~etiquetado] = principal.predict_proba(df.loc[~etiquetado, columnas])[:, 1]

    return {
        "comparacion": pd.DataFrame(comparacion),
        "y_prueba": y_prueba.to_numpy(),
        "prob_prueba": prueba,
        "coeficientes": coeficientes,
        "riesgo": riesgo,
        "variables": columnas,
        "n_entrenamiento": len(X_ent),
        "n_prueba": len(X_prueba),
        "tasa_abandono": float(y.mean()),
    }
