# Proyecto 1: Clasificación de Contaminantes en Líneas de Producción (IA)

## Descripción
Este sistema de visión artificial y aprendizaje automático está diseñado para el control de calidad industrial. Su objetivo es detectar elementos contaminantes (clips y metales) en granos de arroz mediante la clasificación de imágenes binarizadas de $128 \times 128$ píxeles.

---

## Fases del Proyecto

### 1. Preprocesamiento y Vectorización
Utilizando OpenCV, las imágenes se estandarizan mediante:
* **Redimensionamiento:** $128 \times 128$ píxeles.
* **Segmentación:** Conversión a escala de grises y binarización (Fondo=1, Objeto=0).
* **Vectorización:** Generación de vectores de $16,384$ características por muestra.

### 2. Consolidación de Datos
Se integraron cuatro conjuntos de datos grupales para un total de 120 muestras. El proceso incluyó la corrección de polaridad de píxeles, limpieza de etiquetas y aleatorización (*shuffling*) de las muestras para eliminar sesgos de captura. El dataset maestro se ubica en `data/processed/dataset_final_consolidado.csv`.

### 3. Evaluación de Modelos
Se evaluaron los cinco algoritmos fundamentales del curso IE0435 (80% entrenamiento, 20% prueba):

| Algoritmo | Exactitud (Accuracy) |
| :--- | :---: |
| **Árbol de Decisión** (Seleccionado) | **79.2%** |
| **Random Forest** | 70.8% |
| **SVM Lineal** | 62.5% |
| **K-Nearest Neighbors (KNN)** | 62.5% |
| **Naïve Bayes (Gaussian)** | 50.0% |

**Conclusión técnica:** Se seleccionó el **Árbol de Decisión** como modelo final debido a su desempeño superior (79.2%), demostrando una mejor capacidad para extraer reglas morfológicas precisas en este conjunto de datos binarizados frente a modelos de ensamble o lineales.

---

## Estructura del Repositorio
* `data/processed/`: Datasets individuales y consolidado.
* `models/`: Modelos entrenados en formato `.joblib`.
* `scripts/`: Código fuente para procesamiento, integración y entrenamiento.

---

## Guía de Ejecución

1. **Instalar dependencias:**
   `pip install pandas scikit-learn joblib opencv-python`

2. **Unir datasets:**
   `python scripts/unir_datasets.py`

3. **Entrenar y exportar modelos:**
   `python scripts/exportar_modelos.py`

4. **Evaluar métricas:**
   `python scripts/evaluar_modelos.py`

---

## Autor
**Daniel Valverde Ortiz - C28041**
*Escuela de Ingeniería Eléctrica, Universidad de Costa Rica*
