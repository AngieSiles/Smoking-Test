
# Predicción de Tabaquismo — Smoking Dataset 🚬

## Descripción y Objetivo del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de **predecir si un paciente es fumador o no**, basándose en sus características biométricas y resultados de análisis médicos (clasificación binaria).

Dado que en el ámbito de la salud la detección precisa es fundamental y existe un ligero desbalance de clases en los datos originales, la métrica principal elegida para evaluar el desempeño de los modelos es el **F1-Score** sobre la clase positiva (fumador), buscando un equilibrio óptimo entre la precisión y el recall.


## Instrucciones de Ejecución y Entorno

Para reproducir este proyecto en tu máquina local, sigue estos pasos:

### 1. Clonar el repositorio

```bash
git clone https://github.com/AngieSiles/Smoking-Test
cd Smoking-Test
```

### 2. Configurar el entorno virtual

Se recomienda utilizar **Python 3.9 o superior**.

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### 3. Ejecución del código

Ejecutar las notebooks en el siguiente orden desde Jupyter:

| Orden | Notebook | Descripción |
|-------|----------|-------------|
| 1–2 | `01_...` / `02_eda.ipynb` | Ingesta de datos y Análisis Exploratorio (EDA) |
| 3 | `03_procesamiento.ipynb` | Genera el dataset limpio y escalado en `data/processed/` |
| 4 | `04_entrenamiento_y_optimizacion.ipynb` | Entrena los modelos, busca los mejores hiperparámetros y guarda el modelo final en `models/` |
| 5 | `05_validacion.ipynb` | Evalúa el modelo guardado y muestra métricas e importancia de variables |
| 6 | `06_prediccion.ipynb` | Toma datos nuevos de `data/raw/`, aplica el pipeline de inferencia y exporta resultados a `data/predictions/` |


## Estructura del Proyecto

```
Smoking-Test/
│
├── data/
│   ├── raw/                 # Datos originales sin procesar y nuevos datos para inferencia
│   ├── processed/           # Dataset limpio y listo para el modelado (smoking_processed.csv)
│   └── predictions/         # Resultados de las inferencias (nuevas_predicciones.csv)
│
├── models/                  # Artefactos guardados para reproducción y producción
│   ├── modelo_final.joblib
│   ├── feature_names.joblib
│   └── scaler_full.joblib
│
├── notebooks/               # Jupyter Notebooks enumeradas secuencialmente
│   ├── 01_lectura_y_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_procesamiento.ipynb
│   ├── 04_entrenamiento_y_optimizacion.ipynb
│   ├── 05_validacion.ipynb
│   └── 06_prediccion.ipynb
│
└── README.md
```


## Resumen de Experimentos y Decisiones de Desarrollo

### 1. Selección y Limpieza de Features

- Se descartaron variables por analisis con la variable objetivo y sin separación visual en el EDA (`eyesight`, `hearing`, `urine_protein`, `fasting_blood_sugar`, etc.).

### 2. Feature Engineering

Se crearon tres variables con alto poder discriminativo:

| Variable | Composición | Justificación |
|----------|-------------|---------------|
| `liver_score` | `gtp + ast + alt` | Resume el daño hepático asociado al tabaquismo |
| `lipid_ratio` | `triglicéridos / hdl` | Captura el impacto metabólico combinado del tabaco |
| `gender_x_hemoglobin` | interacción género × hemoglobina | Agrupa los dos predictores más fuertes del dataset |

### 3. Escalado y Manejo de Outliers

- Limpieza de outliers mediante el método del **Rango Intercuartílico (IQR)**, eliminando únicamente registros anómalos multivariados (~0.3% del dataset).
- Aplicación de **StandardScaler** sobre todas las variables numéricas continuas (excluyendo binarias como `gender`, `tartar`, `dental_caries`), guardando el objeto `scaler_full.joblib` para garantizar una inferencia matemáticamente idéntica en producción.

### 4. Entrenamiento y Optimización

- **Baselines:** Regresión Logística, Random Forest y XGBoost con `class_weight='balanced'` / `scale_pos_weight` para mitigar el desbalance de clases (63/37).
- **Optimización:** `RandomizedSearchCV` con validación cruzada estratificada (5 folds), métrica **F1-Score** sobre la clase positiva.
- **Modelo ganador:** Random Forest Baseline por mayor F1-Score en test.

#### Resultados completos

| Modelo | F1 | Accuracy | Recall | Precision |
|--------|----|----------|--------|-----------|
| LR Baseline | 0.7087 | 0.7266 | 0.9095 | 0.5805 |
| LR Optimizado | 0.7085 | 0.7268 | 0.9078 | 0.5809 |
| **RF Baseline** | **0.7506** | **0.7962** | **0.8385** | **0.6793** |
| RF Optimizado | 0.7494 | 0.7960 | 0.8343 | 0.6802 |
| XGB Baseline | 0.7191 | 0.7612 | 0.8357 | 0.6310 |
| XGB Optimizado | 0.7372 | 0.7849 | 0.8250 | 0.6662 |

nota!!!
subido el link del models: 
https://drive.google.com/drive/folders/1gxF19ae--2I6pIoopKe_d0XLvNluxLE6


## Conclusiones Principales

**La optimización de hiperparámetros tuvo impacto marginal:** la ganancia fue de +0.018 en XGBoost y prácticamente nula en LR y RF (+0.000 / -0.001), lo que indica que el límite de rendimiento está determinado principalmente por los datos y el feature engineering, no por los hiperparámetros. El RF Baseline superó a su versión optimizada, lo que refuerza esta conclusión.

**Modelo seleccionado — Random Forest Baseline** con F1=0.7506, Accuracy=0.7962, Recall=0.8385 y Precision=0.6793. Ofrece el mejor balance entre precisión y recall: detecta el 84% de los fumadores manteniendo una precisión aceptable.

**Importancia de variables:** las variables fisiológicas más predictivas son el género, el nivel de hemoglobina y los indicadores de función hepática (`liver_score`) y lipídica (`lipid_ratio`), validando el feature engineering realizado.

**Pipeline reproducible:** se construyó un pipeline de inferencia (`aplicar_pipeline`) que automatiza el encoding, la creación de variables derivadas y el escalado, previniendo errores por columnas faltantes o tipos de datos incorrectos al momento de predecir sobre datos nuevos.
