# Retención de alumnos: sistema de decisión con datos

Actividad de Streamlit "Toma de decisiones con datos" (Ciencia de Datos, Universidad Anáhuac).
La app recorre un flujo completo de análisis en seis pestañas, desde entender el problema hasta proponer un plan piloto, con datos **sintéticos y anonimizados** de 250 alumnos.

**Pregunta de decisión:** ¿a qué alumnos conviene apoyar primero para reducir el abandono, y con qué tipo de apoyo?

## Requisitos

- Python 3.10 o superior (probado con 3.15).
- `pip` y, opcionalmente, `git`.
- Dependencias (en `requirements.txt`): `streamlit`, `pandas`, `scikit-learn`. Se probó con streamlit 1.66, pandas 3.0 y scikit-learn 1.9.

## Cómo ejecutarlo

1. Clona el repositorio y entra a la carpeta:

   ```bash
   git clone https://github.com/marcosaw/f05_actividad_streamlit.git
   cd f05_actividad_streamlit
   ```

2. Crea y activa un entorno virtual.

   Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Windows (cmd):

   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   macOS / Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Inicia la app:

   ```bash
   streamlit run app.py
   ```

5. Abre `http://localhost:8501` en el navegador (Streamlit lo abre solo). Para detenerla, presiona `Ctrl + C` en la terminal.

### Problemas frecuentes

- **Tarda en abrir la primera vez:** la primera importación de scikit-learn y Streamlit puede tardar unos minutos en Windows. Las siguientes son rápidas.
- **PowerShell no deja activar el entorno:** ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` o usa `activate.bat` desde cmd.
- **El puerto 8501 está ocupado:** usa `streamlit run app.py --server.port 8502`.
- **El texto no se lee en modo oscuro:** el tema claro está fijado en `.streamlit/config.toml`. Si cambias esa configuración, la app puede verse mal en navegadores con modo oscuro.

## Qué hace cada pestaña

| Pestaña | Archivo | Contenido |
|---|---|---|
| 1. Entender | `tabs/entender.py` | Contexto, pregunta de decisión, variables y vista previa de los datos. |
| 2. Limpiar | `tabs/limpiar.py` | Detecta duplicados, valores fuera de rango y faltantes. Permite conservarlos, convertirlos en faltante o excluirlos, imputa con la mediana, muestra una bitácora y descarga `alumnos_limpios.csv`. |
| 3. Descriptiva | `tabs/descriptiva.py` | Indicadores (KPI) y retención por segmento categórico y por cuartiles de las variables numéricas. |
| 4. Predictiva | `tabs/predictiva.py` | Compara regresión logística y Random Forest, con umbral ajustable, matriz de confusión y coeficientes. |
| 5. Prescriptiva | `tabs/prescriptiva.py` | Clasifica a cada alumno en riesgo alto, medio o bajo y sugiere un apoyo según su motivo principal (pago pendiente, materias reprobadas, asistencia, promedio o participación). |
| 6. Decisiones | `tabs/decisiones.py` | Plan piloto con grupo de control y exportación a `plan_piloto_retencion.csv`. |

La lógica de entrenamiento está en `tabs/modelo.py` y la comparten las pestañas 4, 5 y 6. Si limpiaste los datos en la pestaña 2, el resto de la app usa la versión limpia.

## Datos

Los datos son sintéticos. No corresponden a alumnos reales.

- `data/retencion_alumnos.csv`: versión con errores inyectados a propósito para practicar la limpieza (17 valores faltantes, 3 asistencias mayores a 100, 2 promedios fuera de rango, 1 materias reprobadas negativa, 1 valor de 200 horas de estudio y 3 filas duplicadas; 253 filas en total). **Es la que carga la app.**
- `data/retencion_alumnos_original.csv`: versión limpia de 250 filas, de respaldo.

| Columna | Descripción |
|---|---|
| `id_alumno` | Identificador anónimo (no se usa como variable del modelo). |
| `promedio_anterior` | Promedio del periodo anterior (0 a 10). |
| `porcentaje_asistencia` | Asistencia (0 a 100). |
| `materias_reprobadas` | Número de materias reprobadas. |
| `horas_estudio_semanal` | Horas de estudio por semana. |
| `nivel_socioeconomico` | Bajo, Medio o Alto. |
| `modalidad_estudio` | Presencial u otra modalidad. |
| `pago_al_dia` | Sí / No. |
| `participacion_actividades` | Baja, Media o Alta. |
| `retencion` | 1 = permanece, 0 = abandona. |

## Modelos de machine learning

El objetivo a predecir es el **abandono** (`1 − retencion`), que ocurre en el 30 % de los alumnos.

- **Regresión logística** (modelo principal): `class_weight="balanced"`, regularización L2, variables numéricas estandarizadas. Se puede explicar con coeficientes.
- **Random Forest** (comparación): 300 árboles, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, `random_state=42`.
- **Preparación:** mediana para faltantes numéricos, "(sin dato)" para faltantes categóricos y one-hot para categorías.
- **Evaluación:** 25 % de los alumnos como prueba (estratificado) y validación cruzada de 5 pliegues. AUC en validación cruzada: 0.918 ± 0.017 (logística) y 0.910 ± 0.042 (Random Forest).
- **Niveles de riesgo:** alto con probabilidad ≥ 0.70, medio entre 0.40 y 0.70, bajo por debajo de 0.40. Los umbrales se pueden ajustar en la app.

## Estructura del proyecto

```
app.py                  Punto de entrada: título, carga de datos y las seis pestañas
tabs/                   Una pestaña por archivo, más modelo.py con el entrenamiento
data/                   CSV con errores (el que usa la app) y CSV original limpio
.streamlit/config.toml  Tema claro fijo
requirements.txt        Dependencias
```

## Límites

- Los datos son sintéticos, así que los resultados ilustran el método, no la realidad de una escuela.
- La separación entre entrenamiento y prueba es aleatoria, no temporal.
- La imputación con la mediana de la pestaña 2 se calcula con todos los datos antes de separar la prueba, lo que produce una fuga leve. En producción debe hacerse solo con el entrenamiento.
- Por usar `class_weight`, las probabilidades quedan infladas (riesgo medio de 38 % frente a 30 % real de abandono). Sirven para ordenar alumnos, no como cifras exactas.
- Los patrones indican correlación, no causalidad. Antes de actuar sobre un alumno hay que revisar el caso, cuidar posibles sesgos y proteger sus datos personales.
