# Retención de alumnos: sistema de decisión con datos

Actividad de Streamlit "Toma de decisiones con datos" (Ciencia de Datos, Universidad Anáhuac).
La app guía un flujo completo en seis pestañas, con datos sintéticos y anonimizados de 250 alumnos.

1. **Entender**: el problema y la pregunta de decisión.
2. **Limpiar**: detecta duplicados, valores fuera de rango y faltantes, y los corrige con bitácora.
3. **Descriptiva**: retención por segmento y por cuartiles.
4. **Predictiva**: regresión logística y Random Forest para estimar el riesgo de abandono.
5. **Prescriptiva**: recomendaciones de apoyo por alumno según su motivo principal de riesgo.
6. **Decisiones**: plan piloto con grupo de control y exportación a CSV.

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Datos

- `data/retencion_alumnos.csv`: versión con errores inyectados a propósito (nulos, valores fuera de rango, duplicados) para practicar la limpieza.
- `data/retencion_alumnos_original.csv`: versión limpia, de respaldo.

## Límites

Los datos son sintéticos y la validación usa una partición aleatoria, no temporal. Las probabilidades sirven para ordenar alumnos, no como cifras exactas, y los resultados indican correlación, no causalidad.
