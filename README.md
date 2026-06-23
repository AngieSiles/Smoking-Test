# Predicción de Tabaquismo — Smoking Dataset 🚬

##  Descripción y Objetivo del Proyecto

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

### 3. Instalar las dependencias

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 4. Ejecución del código

 Ejecutar las notebooks en el siguiente orden desde Jupyter:

| Orden | Notebook | Descripción |
|-------|----------|-------------|
| 1–2 | `01_...` / `02_eda.ipynb` | Ingesta de datos y Análisis Exploratorio (EDA) |
| 3 | `03_procesamiento.ipynb` | Genera el dataset limpio y escalado en `data/processed/` |
| 4 | `04_entrenamiento_y_optimizacion.ipynb` | Entrena los modelos, busca los mejores hiperparámetros y guarda el modelo final en `models/` |
| 5 | `05_validacion.ipynb` | Evalúa el modelo guardado y muestra métricas e importancia de variables |
| 6 | `06_prediccion.ipynb` | Toma datos nuevos de `data/raw/`, aplica el pipeline de inferencia y exporta resultados a `data/predictions/` |



##  Estructura del Proyecto

```raw
Smoking-Test/
│
├── data/
│   ├── raw/                 # Datos originales sin procesar y nuevos datos para inferencia
│   ├── processed/           # Dataset limpio y listo para el modelado (smoking_processed.csv)
│   └── predictions/         # Resultados de las inferencias (nuevas_predicciones.csv)
│
├── models/                  # Artefactos guardados para reproducción y producción
│   ├── models_guardado.joblib
│   ├── feature_names.joblib
│   └── scaler_age.joblib
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

- Se descartaron variables con correlación inferior a **0.10** con la variable objetivo y sin separación visual en el EDA (`eyesight`, `hearing`, `urine_protein`, `fasting_blood_sugar`, `tartar`, etc.).
- Se eliminaron variables redundantes o colineales: `weight_kg` y `height_cm` (colineales con `waist_cm`); `cholesterol` y `ldl` (redundantes frente al perfil lipídico combinado).

### 2. Feature Engineering

Se crearon tres variables clave con alto poder discriminativo:

| Variable | Composición | Justificación |
|----------|-------------|---------------|
| `liver_score` | `gtp + ast + alt` | Resume el daño hepático asociado al tabaquismo |
| `lipid_ratio` | `triglicéridos / hdl` | Captura el impacto metabólico combinado del tabaco |
| `gender_x_hemoglobin` | interacción género × hemoglobina | Agrupa los dos predictores más fuertes del dataset |

### 3. Escalado y Manejo de Outliers

- Limpieza de outliers mediante el método del **Rango Intercuartílico (IQR)**, eliminando únicamente registros anómalos multivariados (~0.3% del dataset).
- Aplicación de **StandardScaler** para normalizar la variable de edad, guardando el objeto para garantizar una inferencia matemáticamente idéntica en producción.

### 4. Entrenamiento y Optimización

- **Baselines:** Regresión Logística, Random Forest y XGBoost con `class_weight='balanced'` para mitigar el desbalance de clases (63/37).
- **Optimización:** `RandomizedSearchCV` con validación cruzada estratificada (5 folds).
- **Modelo ganador:** Random Forest Optimizado.

| Modelo | F1-Score | ROC-AUC | Accuracy |
|--------|----------|---------|----------|
| Regresión Logística | Alto recall, baja precisión | — | — |
| **Random Forest (Optimizado)** | **~0.730** | **~0.870** | **~0.780** |
| XGBoost | Competitivo | — | — |

nota!!!
subido el link del models: 
https://drive.google.com/drive/folders/1gxF19ae--2I6pIoopKe_d0XLvNluxLE6


## Conclusiones Principales

**La ingeniería de características fue el factor diferencial:** el preprocesamiento, la eliminación de redundancias y la creación de variables como `liver_score` y `lipid_ratio` tuvieron un impacto más determinante en el rendimiento predictivo que la optimización exhaustiva de hiperparámetros (cuyo aporte marginal fue de **+0.013** en el F1-Score).

**Importancia de variables:** según el modelo Random Forest, las variables fisiológicas más predictivas son el género, el nivel de hemoglobina y los indicadores de función hepática y lipídica.

**Pipeline Reproducible:** se construyó un pipeline de inferencia  que automatiza la creación de variables y el escalado, previniendo errores por columnas faltantes 
